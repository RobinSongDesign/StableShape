using ComputeSharp;
using Rhino.Geometry;
using System;
using System.Collections.Generic;

namespace StableShapeGPU.Properties
{
    /// <summary>
    /// GPU implementation of Jos Stam's "Stable Fluids" on a regular 3D grid.
    /// <para>
    /// Design rule: the simulation state never leaves the GPU. A whole time step is recorded
    /// into a single <see cref="ComputeContext"/> and submitted as one batch, so the CPU waits
    /// for the GPU once per frame instead of once per shader. Data is only downloaded when the
    /// caller explicitly asks for it (<see cref="GetDensity"/>, <see cref="DrawVector"/>).
    /// </para>
    /// </summary>
    public sealed class StableFluid3D : IVelocityField3D, IDisposable
    {
        /// <summary>Jacobi iterations used by the diffusion and the pressure solve.</summary>
        private const int SolverIterations = 20;

        /// <summary>Scale applied to the force lines when they are rasterized into the grid.</summary>
        private const float ForceScale = 0.8f;

        private readonly int sizeX, sizeY, sizeZ;
        private readonly GraphicsDevice gpuDevice;

        // Velocity field and its previous state (swapped by reference, never copied)
        private ReadWriteTexture3D<float> u, v, w;
        private ReadWriteTexture3D<float> u_prev, v_prev, w_prev;

        // External force field, uploaded once and re-applied every step (null when there is none)
        private ReadWriteTexture3D<float> u_force, v_force, w_force;

        private ReadWriteTexture3D<float> density;
        private ReadWriteTexture3D<float> density_prev;

        private ReadWriteTexture3D<float> pressure;
        private ReadWriteTexture3D<float> pressure_prev;

        private ReadWriteTexture3D<float> divergency;

        // Fluid parameters
        private readonly float dt;   // delta time
        private readonly float diff; // diffusion rate
        private readonly float visc; // viscosity

        public StableFluid3D(int width, int height, int depth, float dt = 0.1f, float diff = 0.0001f, float visc = 0.0001f)
            : this(GraphicsDevice.GetDefault(), width, height, depth, dt, diff, visc)
        {
        }

        public StableFluid3D(GraphicsDevice device, int width, int height, int depth, float dt = 0.1f, float diff = 0.0001f, float visc = 0.0001f)
        {
            gpuDevice = device;
            sizeX = width;
            sizeY = height;
            sizeZ = depth;

            this.dt = dt;
            this.diff = diff;
            this.visc = visc;

            u = AllocateField();
            v = AllocateField();
            w = AllocateField();
            u_prev = AllocateField();
            v_prev = AllocateField();
            w_prev = AllocateField();

            density = AllocateField();
            density_prev = AllocateField();

            pressure = AllocateField();
            pressure_prev = AllocateField();
            divergency = AllocateField();
        }

        // --- IVelocityField3D ---
        public int SizeX => sizeX;
        public int SizeY => sizeY;
        public int SizeZ => sizeZ;
        public ReadWriteTexture3D<float> U => u;
        public ReadWriteTexture3D<float> V => v;
        public ReadWriteTexture3D<float> W => w;

        /// <summary>The device that owns every texture of this solver.</summary>
        public GraphicsDevice Device => gpuDevice;

        // ------------------------------------------------------------------
        // Sources (CPU -> GPU). Each call uploads once, whatever the input size.
        // ------------------------------------------------------------------

        /// <summary>Adds an amount of density at every dot.</summary>
        public void AddDot(List<Point3d> dots, float amount)
        {
            if (dots == null || dots.Count == 0) return;

            // Accumulate all the dots on the CPU first, then upload them in one go
            var delta = new float[sizeZ, sizeY, sizeX];
            foreach (var d in dots)
            {
                int i = Clamp((int)d.X, 1, sizeX - 2);
                int j = Clamp((int)d.Y, 1, sizeY - 2);
                int k = Clamp((int)d.Z, 1, sizeZ - 2);
                delta[k, j, i] += amount;
            }

            using (ReadWriteTexture3D<float> deltaGpu = gpuDevice.AllocateReadWriteTexture3D<float>(delta))
            {
                gpuDevice.For(sizeX, sizeY, sizeZ, new AddFieldShader(density, deltaGpu));
            }
        }

        /// <summary>Adds a single amount of density to one cell.</summary>
        public void AddDensity(int x, int y, int z, float amount)
        {
            AddDot(new List<Point3d> { new Point3d(x, y, z) }, amount);
        }

        /// <summary>
        /// Replaces the external force field. The forces stay on the GPU
        /// and are re-applied at the beginning of every step.
        /// </summary>
        public void AddForces(List<Line> forces)
        {
            DisposeForces();
            if (forces == null || forces.Count == 0) return;

            // Rasterize all the forces on the CPU (ComputeSharp expects [z, y, x] ordering)
            var uCpu = new float[sizeZ, sizeY, sizeX];
            var vCpu = new float[sizeZ, sizeY, sizeX];
            var wCpu = new float[sizeZ, sizeY, sizeX];

            foreach (var f in forces)
            {
                int i = Clamp((int)f.From.X, 0, sizeX - 1);
                int j = Clamp((int)f.From.Y, 0, sizeY - 1);
                int k = Clamp((int)f.From.Z, 0, sizeZ - 1);

                uCpu[k, j, i] += (float)(f.To.X - f.From.X) * ForceScale;
                vCpu[k, j, i] += (float)(f.To.Y - f.From.Y) * ForceScale;
                wCpu[k, j, i] += (float)(f.To.Z - f.From.Z) * ForceScale;
            }

            u_force = gpuDevice.AllocateReadWriteTexture3D<float>(uCpu);
            v_force = gpuDevice.AllocateReadWriteTexture3D<float>(vCpu);
            w_force = gpuDevice.AllocateReadWriteTexture3D<float>(wCpu);
        }

        // ------------------------------------------------------------------
        // Simulation (GPU only)
        // ------------------------------------------------------------------

        /// <summary>Runs one full time step as a single GPU batch.</summary>
        public void Update()
        {
            using (ComputeContext context = gpuDevice.CreateComputeContext())
            {
                RecordStep(in context);
            }
        }

        /// <summary>
        /// Records one full time step into an existing context without executing it.
        /// Lets the caller chain further GPU work (e.g. the mesh solver) into the same batch.
        /// </summary>
        public void RecordStep(in ComputeContext context)
        {
            ApplyForces(in context);
            VelocityStep(in context);
            DensityStep(in context);
        }

        private void ApplyForces(in ComputeContext context)
        {
            if (u_force == null) return;

            context.For(sizeX, sizeY, sizeZ, new AddFieldShader(u, u_force));
            context.For(sizeX, sizeY, sizeZ, new AddFieldShader(v, v_force));
            context.For(sizeX, sizeY, sizeZ, new AddFieldShader(w, w_force));
            context.Barrier(u);
            context.Barrier(v);
            context.Barrier(w);
        }

        private void VelocityStep(in ComputeContext context)
        {
            // Swap velocity
            (u, u_prev) = (u_prev, u);
            (v, v_prev) = (v_prev, v);
            (w, w_prev) = (w_prev, w);

            Diffuse(in context, 1, u, u_prev, visc);
            Diffuse(in context, 2, v, v_prev, visc);
            Diffuse(in context, 3, w, w_prev, visc);

            // Project to make sure div is 0
            Project(in context);

            // Swap velocity
            (u, u_prev) = (u_prev, u);
            (v, v_prev) = (v_prev, v);
            (w, w_prev) = (w_prev, w);

            // Advect - fluid transit itself
            Advect(in context, 1, u, u_prev, u_prev, v_prev, w_prev);
            Advect(in context, 2, v, v_prev, u_prev, v_prev, w_prev);
            Advect(in context, 3, w, w_prev, u_prev, v_prev, w_prev);

            // Project to make sure div is 0
            Project(in context);
        }

        private void DensityStep(in ComputeContext context)
        {
            // Swap density
            (density, density_prev) = (density_prev, density);

            // Diffuse
            Diffuse(in context, 0, density, density_prev, diff);

            // Swap back
            (density, density_prev) = (density_prev, density);

            // Advect
            Advect(in context, 0, density, density_prev, u, v, w);
        }

        /// <summary>
        /// Applies the boundary condition of a field (faces, edges and corners) fully on the GPU.
        /// typeIndex: 1 for X velocity; 2 for Y velocity; 3 for Z velocity; 0 for scalar fields.
        /// </summary>
        private void SetBoundary(in ComputeContext context, int typeIndex, ReadWriteTexture3D<float> f)
        {
            context.For(sizeX, sizeY, sizeZ, new SetBoundaryShader(f, typeIndex, sizeX, sizeY, sizeZ));
            context.Barrier(f);
        }

        private void Diffuse(in ComputeContext context, int typeIndex, ReadWriteTexture3D<float> f, ReadWriteTexture3D<float> f0, float rate)
        {
            float a = dt * rate * sizeX * sizeY * sizeZ;
            for (int n = 0; n < SolverIterations; n++)
            {
                context.For(sizeX, sizeY, sizeZ, new DiffuseShader(f, f0, a, sizeX, sizeY, sizeZ));
                context.Barrier(f);
                SetBoundary(in context, typeIndex, f);
            }
        }

        /// <summary>
        /// Core method: advects a physics value through the velocity field.
        /// d is the target field, d0 is the current field.
        /// </summary>
        private void Advect(in ComputeContext context, int typeIndex, ReadWriteTexture3D<float> d, ReadWriteTexture3D<float> d0,
            ReadWriteTexture3D<float> velU, ReadWriteTexture3D<float> velV, ReadWriteTexture3D<float> velW)
        {
            context.For(sizeX, sizeY, sizeZ, new AdvectShader(d, d0, velU, velV, velW, dt, sizeX, sizeY, sizeZ));
            context.Barrier(d);
            SetBoundary(in context, typeIndex, d);
        }

        /// <summary>Makes the velocity field divergence free.</summary>
        private void Project(in ComputeContext context)
        {
            // Divergence of the velocity field
            context.For(sizeX, sizeY, sizeZ, new DivergenceShader(u, v, w, divergency, sizeX, sizeY, sizeZ));
            context.Barrier(divergency);
            SetBoundary(in context, 0, divergency);

            // Start the pressure solve from zero
            context.Clear(pressure);
            context.Clear(pressure_prev);

            // Solve the Poisson equation with ping-pong Jacobi iterations
            for (int n = 0; n < SolverIterations; n++)
            {
                context.For(sizeX, sizeY, sizeZ, new PressureSolverShader(pressure_prev, pressure, divergency, sizeX, sizeY, sizeZ));
                context.Barrier(pressure);
                SetBoundary(in context, 0, pressure);
                (pressure, pressure_prev) = (pressure_prev, pressure);
            }

            // The latest solution is in pressure_prev after the last swap
            context.For(sizeX, sizeY, sizeZ, new VelocityCorrectionShader(u, v, w, pressure_prev, sizeX, sizeY, sizeZ));
            context.Barrier(u);
            context.Barrier(v);
            context.Barrier(w);
            SetBoundary(in context, 1, u);
            SetBoundary(in context, 2, v);
            SetBoundary(in context, 3, w);
        }

        // ------------------------------------------------------------------
        // Read back (GPU -> CPU). Only call these when the data is really needed.
        // ------------------------------------------------------------------

        /// <summary>Downloads the density field, indexed as [z, y, x].</summary>
        public float[,,] GetDensity()
        {
            return density.ToArray();
        }

        /// <summary>Downloads the velocity field and returns one line per cell.</summary>
        public List<Line> DrawVector()
        {
            // ComputeSharp returns 3D textures as [z, y, x]
            float[,,] uCpu = u.ToArray();
            float[,,] vCpu = v.ToArray();
            float[,,] wCpu = w.ToArray();

            var lns = new List<Line>(sizeX * sizeY * sizeZ);
            for (int i = 0; i < sizeX; i++)
            {
                for (int j = 0; j < sizeY; j++)
                {
                    for (int k = 0; k < sizeZ; k++)
                    {
                        var vec = new Vector3d(uCpu[k, j, i], vCpu[k, j, i], wCpu[k, j, i]);
                        lns.Add(new Line(new Point3d(i, j, k), vec));
                    }
                }
            }
            return lns;
        }

        // ------------------------------------------------------------------
        // Helpers
        // ------------------------------------------------------------------

        /// <summary>Allocates a zero-initialized scalar field.</summary>
        private ReadWriteTexture3D<float> AllocateField()
        {
            return gpuDevice.AllocateReadWriteTexture3D<float>(sizeX, sizeY, sizeZ, AllocationMode.Clear);
        }

        private void DisposeForces()
        {
            u_force?.Dispose();
            v_force?.Dispose();
            w_force?.Dispose();
            u_force = v_force = w_force = null;
        }

        /// <summary>Releases every GPU texture owned by the solver.</summary>
        public void Dispose()
        {
            DisposeForces();
            u.Dispose(); v.Dispose(); w.Dispose();
            u_prev.Dispose(); v_prev.Dispose(); w_prev.Dispose();
            density.Dispose(); density_prev.Dispose();
            pressure.Dispose(); pressure_prev.Dispose();
            divergency.Dispose();
        }

        public static int Clamp(int a, int b, int c)
        {
            if (a < b) return b;
            if (a > c) return c;
            return a;
        }
    }

    /// <summary>One relaxation sweep of the implicit diffusion, interior cells only.</summary>
    [AutoConstructor]
    [EmbeddedBytecode(DispatchAxis.XYZ)]
    public readonly partial struct DiffuseShader : IComputeShader
    {
        private readonly ReadWriteTexture3D<float> f;
        private readonly ReadWriteTexture3D<float> f0;
        private readonly float a;
        private readonly int sizeX, sizeY, sizeZ;

        public void Execute()
        {
            int i = ThreadIds.X;
            int j = ThreadIds.Y;
            int k = ThreadIds.Z;

            // Boundary cells are owned by SetBoundaryShader
            if (i > 0 && i < sizeX - 1 && j > 0 && j < sizeY - 1 && k > 0 && k < sizeZ - 1)
            {
                f[i, j, k] = (f0[i, j, k] + a * (f[i - 1, j, k] + f[i + 1, j, k]
                                               + f[i, j - 1, k] + f[i, j + 1, k]
                                               + f[i, j, k - 1] + f[i, j, k + 1])) / (1 + 6 * a);
            }
        }
    }

    /// <summary>Semi-Lagrangian advection with trilinear interpolation.</summary>
    [AutoConstructor]
    [EmbeddedBytecode(DispatchAxis.XYZ)]
    public readonly partial struct AdvectShader : IComputeShader
    {
        private readonly ReadWriteTexture3D<float> d;
        private readonly ReadWriteTexture3D<float> d0;
        private readonly ReadWriteTexture3D<float> u, v, w;
        private readonly float dt;
        private readonly int sizeX, sizeY, sizeZ;

        public void Execute()
        {
            int i = ThreadIds.X;
            int j = ThreadIds.Y;
            int k = ThreadIds.Z;

            // Only process the interior cells
            if (i > 0 && i < sizeX - 1 && j > 0 && j < sizeY - 1 && k > 0 && k < sizeZ - 1)
            {
                float dt0 = dt * sizeX;

                // Trace back to the source point
                float x = i - dt0 * u[ThreadIds.XYZ];
                float y = j - dt0 * v[ThreadIds.XYZ];
                float z = k - dt0 * w[ThreadIds.XYZ];

                // Keep the source point inside the grid
                x = Hlsl.Max(0.5f, Hlsl.Min(sizeX - 1.5f, x));
                y = Hlsl.Max(0.5f, Hlsl.Min(sizeY - 1.5f, y));
                z = Hlsl.Max(0.5f, Hlsl.Min(sizeZ - 1.5f, z));

                // Surrounding cells
                int i0 = (int)x;
                int i1 = i0 + 1;
                int j0 = (int)y;
                int j1 = j0 + 1;
                int k0 = (int)z;
                int k1 = k0 + 1;

                // Interpolation weights
                float s1 = x - i0;
                float t1 = y - j0;
                float r1 = z - k0;

                // Trilinear interpolation: X, then Y, then Z
                float d_x0 = Hlsl.Lerp(d0[i0, j0, k0], d0[i1, j0, k0], s1);
                float d_x1 = Hlsl.Lerp(d0[i0, j1, k0], d0[i1, j1, k0], s1);
                float d_x2 = Hlsl.Lerp(d0[i0, j0, k1], d0[i1, j0, k1], s1);
                float d_x3 = Hlsl.Lerp(d0[i0, j1, k1], d0[i1, j1, k1], s1);

                float d_y0 = Hlsl.Lerp(d_x0, d_x1, t1);
                float d_y1 = Hlsl.Lerp(d_x2, d_x3, t1);

                d[ThreadIds.XYZ] = Hlsl.Lerp(d_y0, d_y1, r1);
            }
        }
    }

    [AutoConstructor]
    [EmbeddedBytecode(DispatchAxis.XYZ)]
    public readonly partial struct DivergenceShader : IComputeShader
    {
        private readonly ReadWriteTexture3D<float> u, v, w;
        private readonly ReadWriteTexture3D<float> div;
        private readonly int sizeX, sizeY, sizeZ;

        public void Execute()
        {
            int i = ThreadIds.X;
            int j = ThreadIds.Y;
            int k = ThreadIds.Z;

            if (i > 0 && i < sizeX - 1 && j > 0 && j < sizeY - 1 && k > 0 && k < sizeZ - 1)
            {
                div[ThreadIds.XYZ] = -0.5f * (
                    (u[i + 1, j, k] - u[i - 1, j, k]) / sizeX +
                    (v[i, j + 1, k] - v[i, j - 1, k]) / sizeY +
                    (w[i, j, k + 1] - w[i, j, k - 1]) / sizeZ);
            }
            else
            {
                div[ThreadIds.XYZ] = 0;
            }
        }
    }

    /// <summary>One Jacobi iteration of the pressure Poisson equation (reads p_in, writes p_out).</summary>
    [AutoConstructor]
    [EmbeddedBytecode(DispatchAxis.XYZ)]
    public readonly partial struct PressureSolverShader : IComputeShader
    {
        private readonly ReadWriteTexture3D<float> p_in;
        private readonly ReadWriteTexture3D<float> p_out;
        private readonly ReadWriteTexture3D<float> div;
        private readonly int sizeX, sizeY, sizeZ;

        public void Execute()
        {
            int i = ThreadIds.X;
            int j = ThreadIds.Y;
            int k = ThreadIds.Z;

            // Only compute the interior cells
            if (i > 0 && i < sizeX - 1 && j > 0 && j < sizeY - 1 && k > 0 && k < sizeZ - 1)
            {
                p_out[ThreadIds.XYZ] = (div[ThreadIds.XYZ] +
                                      p_in[i - 1, j, k] + p_in[i + 1, j, k] +
                                      p_in[i, j - 1, k] + p_in[i, j + 1, k] +
                                      p_in[i, j, k - 1] + p_in[i, j, k + 1]) / 6.0f;
            }
        }
    }

    /// <summary>Subtracts the pressure gradient from the velocity field.</summary>
    [AutoConstructor]
    [EmbeddedBytecode(DispatchAxis.XYZ)]
    public readonly partial struct VelocityCorrectionShader : IComputeShader
    {
        private readonly ReadWriteTexture3D<float> u, v, w;
        private readonly ReadWriteTexture3D<float> p;
        private readonly int sizeX, sizeY, sizeZ;

        public void Execute()
        {
            int i = ThreadIds.X;
            int j = ThreadIds.Y;
            int k = ThreadIds.Z;

            // Only compute the interior cells
            if (i > 0 && i < sizeX - 1 && j > 0 && j < sizeY - 1 && k > 0 && k < sizeZ - 1)
            {
                u[ThreadIds.XYZ] -= 0.5f * (p[i + 1, j, k] - p[i - 1, j, k]) * sizeX;
                v[ThreadIds.XYZ] -= 0.5f * (p[i, j + 1, k] - p[i, j - 1, k]) * sizeY;
                w[ThreadIds.XYZ] -= 0.5f * (p[i, j, k + 1] - p[i, j, k - 1]) * sizeZ;
            }
        }
    }

    /// <summary>
    /// Boundary condition for faces, edges and corners in a single race-free pass.
    /// <para>
    /// Every boundary cell copies its nearest interior cell. A face cell flips the sign when the
    /// field is the velocity component normal to that face (no flow through the walls). Edge and
    /// corner cells touch several faces, so they take the average of those face rules, which is
    /// exactly "the average of the adjacent boundary cells" of the reference CPU implementation.
    /// Only interior cells are read and only boundary cells are written, so no CPU fix-up is needed.
    /// </para>
    /// </summary>
    [AutoConstructor]
    [EmbeddedBytecode(DispatchAxis.XYZ)]
    public readonly partial struct SetBoundaryShader : IComputeShader
    {
        private readonly ReadWriteTexture3D<float> f;
        private readonly int typeIndex; // 1 for u, 2 for v, 3 for w, 0 for others
        private readonly int sizeX, sizeY, sizeZ;

        public void Execute()
        {
            int i = ThreadIds.X;
            int j = ThreadIds.Y;
            int k = ThreadIds.Z;

            // Nearest interior cell
            int ci = Hlsl.Max(1, Hlsl.Min(sizeX - 2, i));
            int cj = Hlsl.Max(1, Hlsl.Min(sizeY - 2, j));
            int ck = Hlsl.Max(1, Hlsl.Min(sizeZ - 2, k));

            bool onX = ci != i;
            bool onY = cj != j;
            bool onZ = ck != k;

            if (onX || onY || onZ)
            {
                float signSum = 0;
                float faceCount = 0;

                if (onX)
                {
                    signSum += (typeIndex == 1) ? -1.0f : 1.0f;
                    faceCount += 1.0f;
                }
                if (onY)
                {
                    signSum += (typeIndex == 2) ? -1.0f : 1.0f;
                    faceCount += 1.0f;
                }
                if (onZ)
                {
                    signSum += (typeIndex == 3) ? -1.0f : 1.0f;
                    faceCount += 1.0f;
                }

                f[i, j, k] = f[ci, cj, ck] * (signSum / faceCount);
            }
        }
    }

    /// <summary>targetField += valuesToAdd</summary>
    [AutoConstructor]
    [EmbeddedBytecode(DispatchAxis.XYZ)]
    public readonly partial struct AddFieldShader : IComputeShader
    {
        private readonly ReadWriteTexture3D<float> targetField;
        private readonly ReadWriteTexture3D<float> valuesToAdd;

        public void Execute()
        {
            targetField[ThreadIds.XYZ] += valuesToAdd[ThreadIds.XYZ];
        }
    }


    public class StableFluid2D
    {
        private int sizeX;
        private int sizeY;

        private Mesh mesh;

        // use 2d array to store the field
        private double[,] u;      // X 
        private double[,] v;      // Y
        private double[,] u_prev;
        private double[,] v_prev;
        private double[,] density;
        private double[,] density_prev;

        // fluid parameters
        private double dt;  // delta time
        private double diff; // diffusion rate
        private double visc; // viscosity

        public StableFluid2D(int width, int height, double dt = 0.1f, double diff = 0.0001f, double visc = 0.0001f)
        {
            sizeX = width;
            sizeY = height;

            // initialize the 2d array
            u = new double[sizeX, sizeY];
            v = new double[sizeX, sizeY];
            u_prev = new double[sizeX, sizeY];
            v_prev = new double[sizeX, sizeY];
            density = new double[sizeX, sizeY];
            density_prev = new double[sizeX, sizeY];

            this.dt = dt;
            this.diff = diff;
            this.visc = visc;
        }

        //////////////
        public void AddForce(IList<Line> forces)
        {
            foreach (var f in forces)
            {
                double deltau = f.ToX - f.FromX;
                double deltav = f.ToY - f.FromY;

                AddVelocity((int)f.FromX, (int)f.FromY, deltau, deltav);
            }
        }
        public void AddDot(IList<Point3d> dots, double amount)
        {
            foreach (var d in dots)
            {
                AddDensity((int)d.X, (int)d.Y, amount);
            }
        }
        public void AddVelocity(int x, int y, double amountX, double amountY)
        {
            int i = Clamp(x, 1, sizeX - 2);
            int j = Clamp(y, 1, sizeY - 2);

            u[i, j] += amountX;
            v[i, j] += amountY;
        }

        public void AddDensity(int x, int y, double amount)
        {
            int i = Clamp(x, 1, sizeX - 2);
            int j = Clamp(y, 1, sizeY - 2);

            density[i, j] += amount;
        }

        private void SetBoundary(int typeIndex, double[,] f)
        {
            for (int i = 1; i < sizeX - 1; i++)
            {
                //when the typeIndex is 2, process the velocity on y direction
                if (typeIndex == 2)
                {
                    f[i, 0] = -f[i, 1];
                    f[i, sizeY - 1] = -f[i, sizeY - 2];
                }
                //other cases just set the boudary velociy to be the adjacent one (0 or 1)
                else
                {
                    f[i, 0] = f[i, 1];
                    f[i, sizeY - 1] = f[i, sizeY - 2];
                }
            }

            for (int j = 1; j < sizeY - 1; j++)
            {
                //when the typeIndex is 1, process the velocity on x direction
                if (typeIndex == 1)
                {
                    f[0, j] = -f[1, j];
                    f[sizeX - 1, j] = -f[sizeX - 2, j];
                }
                //other cases just set the boudary velociy to be the adjacent one (0 or 2)
                else
                {
                    f[0, j] = f[1, j];
                    f[sizeX - 1, j] = f[sizeX - 2, j];
                }
            }

            // Explicitly set the corner by average adjacent points
            f[0, 0] = 0.5f * (f[1, 0] + f[0, 1]);
            f[0, sizeY - 1] = 0.5f * (f[1, sizeY - 1] + f[0, sizeY - 2]);
            f[sizeX - 1, 0] = 0.5f * (f[sizeX - 2, 0] + f[sizeX - 1, 1]);
            f[sizeX - 1, sizeY - 1] = 0.5f * (f[sizeX - 2, sizeY - 1] + f[sizeX - 1, sizeY - 2]);
        }

        private void Diffuse(int typeIndex, double[,] f, double[,] f0, double rate)
        {
            double a = dt * rate * sizeX * sizeY;
            for (int k = 0; k < 20; k++)
            {
                for (int i = 1; i < sizeX - 1; i++)
                {
                    for (int j = 1; j < sizeY - 1; j++)
                    {
                        f[i, j] = (f0[i, j] + a * (f[i - 1, j] + f[i + 1, j] +
                                                f[i, j - 1] + f[i, j + 1])) / (1 + 4 * a);
                    }
                }
                SetBoundary(typeIndex, f);
            }
        }

        //Core Method
        //This method is to advect the physics value through the velocity field
        //d is the target array, d0 is the current array
        private void Advect(int typeIndex, double[,] d, double[,] d0, double[,] u, double[,] v)
        {
            double dt0 = dt * sizeX;

            for (int i = 1; i < sizeX - 1; i++)
            {
                for (int j = 1; j < sizeY - 1; j++)
                {
                    double x = i - dt0 * u[i, j];
                    double y = j - dt0 * v[i, j];

                    //ensure the point within the boundary
                    x = Math.Max(0.5f, Math.Min(sizeX - 1.5f, x));
                    y = Math.Max(0.5f, Math.Min(sizeY - 1.5f, y));

                    //Find points surrounding
                    int i0 = (int)x;
                    int i1 = i0 + 1;
                    int j0 = (int)y;
                    int j1 = j0 + 1;

                    //Calculate the interpolation weight
                    double s1 = x - i0;
                    double s0 = 1 - s1;
                    double t1 = y - j0;
                    double t0 = 1 - t1;

                    // Get new value
                    d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                     s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]);
                }
            }
            SetBoundary(typeIndex, d);
        }

        //Project is to make sure there is no divergency in the velocity field
        private void Project(double[,] u, double[,] v, double[,] p, double[,] div)
        {
            //calculate the div of velocity field
            for (int i = 1; i < sizeX - 1; i++)
            {
                for (int j = 1; j < sizeY - 1; j++)
                {
                    div[i, j] = -0.5f * ((u[i + 1, j] - u[i - 1, j]) / sizeX +
                                        (v[i, j + 1] - v[i, j - 1]) / sizeY);
                    p[i, j] = 0;
                }
            }

            SetBoundary(0, div);
            SetBoundary(0, p);

            //solve Poisson equation
            for (int k = 0; k < 20; k++)
            {
                for (int i = 1; i < sizeX - 1; i++)
                {
                    for (int j = 1; j < sizeY - 1; j++)
                    {
                        p[i, j] = (div[i, j] + p[i - 1, j] + p[i + 1, j] + p[i, j - 1] + p[i, j + 1]) / 4;
                    }
                }
            }
            SetBoundary(0, p);

            //Use the Force delta to adjust the velocity field
            for (int i = 1; i < sizeX - 1; i++)
            {
                for (int j = 1; j < sizeY - 1; j++)
                {
                    u[i, j] -= 0.5f * sizeX * (p[i + 1, j] - p[i - 1, j]);
                    v[i, j] -= 0.5f * sizeY * (p[i, j + 1] - p[i, j - 1]);
                }
            }
            SetBoundary(1, u);
            SetBoundary(2, v);
        }

        public void Update()
        {
            VelocityStep();
            DensityStep();
        }

        private void VelocityStep()
        {
            // Swap velocity
            Swap(ref u, ref u_prev);
            Swap(ref v, ref v_prev);

            Diffuse(1, u, u_prev, visc);
            Diffuse(2, v, v_prev, visc);

            // Project to make sure div is 0  
            Project(u, v, u_prev, v_prev);

            // Swap velocity
            Swap(ref u, ref u_prev);
            Swap(ref v, ref v_prev);

            // Advect - fluid transit itself 
            Advect(1, u, u_prev, u_prev, v_prev);
            Advect(2, v, v_prev, u_prev, v_prev);

            // Project to make sure div is 0   
            Project(u, v, u_prev, v_prev);
        }

        private void DensityStep()
        {
            // Swap density
            Swap(ref density, ref density_prev);

            // diffuse  
            Diffuse(0, density, density_prev, diff);

            // Swap back
            Swap(ref density, ref density_prev);

            // Advect  
            Advect(0, density, density_prev, u, v);
        }

        private void Swap(ref double[,] a, ref double[,] b)
        {
            double[,] temp = a;
            a = b;
            b = temp;
        }

        public double[,] GetDensity()
        {
            return density;
        }

        public Mesh GetMesh()
        {
            return mesh;
        }

        public List<Line> DrawVector()
        {
            List<Line> lines = new List<Line>();
            Vector3d[,] vecs = new Vector3d[sizeX, sizeY];
            for (int i = 0; i < sizeX; i++)
            {
                for (int j = 0; j < sizeY; j++)
                {
                    vecs[i, j] = new Vector3d(u[i, j], v[i, j], 0);
                    Line ln = new Line(new Point3d(i, j, 0), vecs[i, j]);
                    lines.Add(ln);
                }
            }
            return lines;
        }

        public int Clamp(int a, int b, int c)
        {
            if (a < b) return b;
            if (a > c) return c;
            return a;
        }
    }
}
