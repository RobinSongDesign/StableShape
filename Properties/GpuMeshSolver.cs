using ComputeSharp;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;

namespace StableShapeGPU.Properties
{
    // Use the vector types from ComputeSharp for GPU data
    using float3 = ComputeSharp.Float3;
    using int2 = ComputeSharp.Int2;

    /// <summary>
    /// Mass-spring mesh driven by a GPU velocity field.
    /// <para>
    /// The particles live on the GPU for the whole simulation. Any number of sub steps can be
    /// recorded into one <see cref="ComputeContext"/>; the vertices are downloaded only when
    /// <see cref="ReconstructMesh"/> is called, i.e. once per displayed frame.
    /// </para>
    /// </summary>
    public sealed class GpuMeshSolver : IDisposable
    {
        private readonly GraphicsDevice gpuDevice;

        // --- GPU Buffers ---
        // Particle data
        private readonly ReadWriteBuffer<float3> particlePositions;
        private readonly ReadWriteBuffer<float3> particleVelocities;
        private readonly ReadWriteBuffer<float3> particleAccelerations;

        // Spring adjacency in CSR layout (read-only on the GPU):
        // particle p owns the entries [adjacencyRanges[p].X, adjacencyRanges[p].Y)
        // of adjacencyNeighbours / adjacencyRestLengths.
        private readonly ReadOnlyBuffer<int2> adjacencyRanges;
        private readonly ReadOnlyBuffer<int> adjacencyNeighbours;
        private readonly ReadOnlyBuffer<float> adjacencyRestLengths;

        // Mesh topology and CPU staging memory for the reconstruction (allocated once)
        private readonly MeshFace[] meshFaces;
        private readonly float3[] positionsCpu;
        private readonly Point3f[] verticesCpu;

        public GpuMeshSolver(Mesh mesh) : this(GraphicsDevice.GetDefault(), mesh)
        {
        }

        public GpuMeshSolver(GraphicsDevice device, Mesh mesh)
        {
            gpuDevice = device;

            int particleCount = mesh.Vertices.Count;

            // --- Particles ---
            positionsCpu = new float3[particleCount];
            verticesCpu = new Point3f[particleCount];
            for (int i = 0; i < particleCount; i++)
            {
                Point3f p = mesh.Vertices[i];
                positionsCpu[i] = new float3(p.X, p.Y, p.Z);
            }

            // --- Springs: one per unique mesh edge ---
            var springs = new List<Spring>();
            var processedEdges = new HashSet<long>();
            foreach (MeshFace face in mesh.Faces)
            {
                AddSpring(springs, processedEdges, face.A, face.B);
                AddSpring(springs, processedEdges, face.B, face.C);
                if (face.IsQuad)
                {
                    AddSpring(springs, processedEdges, face.C, face.D);
                    AddSpring(springs, processedEdges, face.D, face.A);
                }
                else
                {
                    AddSpring(springs, processedEdges, face.C, face.A);
                }
            }

            // --- Upload to GPU ---
            BuildAdjacency(particleCount, springs, out int2[] ranges, out int[] neighbours, out float[] restLengths);

            particlePositions = gpuDevice.AllocateReadWriteBuffer<float3>(positionsCpu);
            particleVelocities = gpuDevice.AllocateReadWriteBuffer<float3>(particleCount, AllocationMode.Clear);
            particleAccelerations = gpuDevice.AllocateReadWriteBuffer<float3>(particleCount, AllocationMode.Clear);
            adjacencyRanges = gpuDevice.AllocateReadOnlyBuffer<int2>(ranges);
            adjacencyNeighbours = gpuDevice.AllocateReadOnlyBuffer<int>(neighbours);
            adjacencyRestLengths = gpuDevice.AllocateReadOnlyBuffer<float>(restLengths);

            // Keep mesh faces on CPU for final reconstruction
            meshFaces = mesh.Faces.ToArray();
        }

        /// <summary>Registers the edge (v0, v1) as a spring unless it already exists.</summary>
        private void AddSpring(List<Spring> springs, HashSet<long> processedEdges, int v0, int v1)
        {
            long edgeId = v0 < v1 ? ((long)v0 << 32) | (uint)v1 : ((long)v1 << 32) | (uint)v0;
            if (!processedEdges.Add(edgeId)) return;

            float3 delta = positionsCpu[v1] - positionsCpu[v0];
            float restlen = (float)Math.Sqrt(delta.X * delta.X + delta.Y * delta.Y + delta.Z * delta.Z);
            springs.Add(new Spring(v0, v1, restlen));
        }

        /// <summary>
        /// Converts the spring list into a per-particle adjacency (CSR layout), so each particle
        /// only visits its own springs on the GPU instead of scanning all of them.
        /// </summary>
        private static void BuildAdjacency(int particleCount, List<Spring> springs, out int2[] ranges, out int[] neighbours, out float[] restLengths)
        {
            // Count the springs attached to each particle
            var counts = new int[particleCount];
            foreach (Spring s in springs)
            {
                counts[s.vertex0]++;
                counts[s.vertex1]++;
            }

            // Prefix sum -> [start, end) of each particle
            ranges = new int2[particleCount];
            var cursor = new int[particleCount];
            int total = 0;
            for (int i = 0; i < particleCount; i++)
            {
                ranges[i] = new int2(total, total + counts[i]);
                cursor[i] = total;
                total += counts[i];
            }

            // Every spring is stored twice, once for each end.
            // At least one element is allocated: GPU buffers cannot be empty.
            neighbours = new int[Math.Max(1, total)];
            restLengths = new float[Math.Max(1, total)];
            foreach (Spring s in springs)
            {
                neighbours[cursor[s.vertex0]] = s.vertex1;
                restLengths[cursor[s.vertex0]++] = s.restlen;

                neighbours[cursor[s.vertex1]] = s.vertex0;
                restLengths[cursor[s.vertex1]++] = s.restlen;
            }
        }

        /// <summary>Runs the given number of sub steps as a single GPU batch.</summary>
        public void Simulate(IVelocityField3D field, int subSteps = 1)
        {
            using (ComputeContext context = gpuDevice.CreateComputeContext())
            {
                RecordSteps(in context, field, subSteps);
            }
        }

        /// <summary>
        /// Records the given number of sub steps into an existing context without executing it,
        /// so the mesh can share one GPU submission with the fluid solver.
        /// </summary>
        public void RecordSteps(in ComputeContext context, IVelocityField3D field, int subSteps = 1)
        {
            int count = particlePositions.Length;

            for (int step = 0; step < subSteps; step++)
            {
                // 1. Take the fluid velocity and move the particles with it
                context.For(
                    count,
                    new MeshAdvectionShader(
                        particlePositions,
                        particleVelocities,
                        field.U, field.V, field.W,
                        field.SizeX, field.SizeY, field.SizeZ,
                        Particle.timeStep,
                        Particle.damping,
                        Particle.drag));
                context.Barrier(particlePositions);
                context.Barrier(particleVelocities);

                // 2. Spring tension of every particle
                context.For(
                    count,
                    new SpringForcesShader(
                        particlePositions,
                        particleAccelerations,
                        adjacencyRanges,
                        adjacencyNeighbours,
                        adjacencyRestLengths,
                        Particle.k,
                        Particle.mass));
                context.Barrier(particleAccelerations);

                // 3. Move the particles with the spring tension
                context.For(
                    count,
                    new ParticleMoveShader(
                        particlePositions,
                        particleVelocities,
                        particleAccelerations,
                        Particle.timeStep,
                        Particle.damping,
                        Particle.drag));
                context.Barrier(particlePositions);
                context.Barrier(particleVelocities);
            }
        }

        /// <summary>
        /// Downloads the particle positions and rebuilds the Rhino mesh.
        /// This is the only GPU -> CPU transfer of the solver: call it once per displayed frame.
        /// </summary>
        public Mesh ReconstructMesh()
        {
            particlePositions.CopyTo(positionsCpu);

            for (int i = 0; i < positionsCpu.Length; i++)
            {
                float3 p = positionsCpu[i];
                verticesCpu[i] = new Point3f(p.X, p.Y, p.Z);
            }

            Mesh remesh = new Mesh();
            remesh.Vertices.AddVertices(verticesCpu);
            remesh.Faces.AddFaces(meshFaces);
            remesh.Normals.ComputeNormals();

            return remesh;
        }

        /// <summary>Releases every GPU buffer owned by the solver.</summary>
        public void Dispose()
        {
            particlePositions.Dispose();
            particleVelocities.Dispose();
            particleAccelerations.Dispose();
            adjacencyRanges.Dispose();
            adjacencyNeighbours.Dispose();
            adjacencyRestLengths.Dispose();
        }
    }

    /// <summary>
    /// Samples the fluid velocity at each particle (trilinear interpolation)
    /// and moves the particle with it.
    /// </summary>
    [AutoConstructor]
    [EmbeddedBytecode(DispatchAxis.X)]
    public readonly partial struct MeshAdvectionShader : IComputeShader
    {
        private readonly ReadWriteBuffer<float3> positions;
        private readonly ReadWriteBuffer<float3> velocities;
        private readonly ReadWriteTexture3D<float> u, v, w;
        private readonly int sizeX, sizeY, sizeZ;
        private readonly float timeStep, damping, drag;

        public void Execute()
        {
            int i = ThreadIds.X; // The index of the current particle
            float3 pos = positions[i];

            // 1. Clamp the sample position to the valid grid bounds for interpolation
            float x = Hlsl.Max(0.5f, Hlsl.Min(sizeX - 1.5f, pos.X));
            float y = Hlsl.Max(0.5f, Hlsl.Min(sizeY - 1.5f, pos.Y));
            float z = Hlsl.Max(0.5f, Hlsl.Min(sizeZ - 1.5f, pos.Z));

            // 2. Integer indices of the 8 surrounding grid cells
            int i0 = (int)x;
            int i1 = i0 + 1;
            int j0 = (int)y;
            int j1 = j0 + 1;
            int k0 = (int)z;
            int k1 = k0 + 1;

            // 3. Fractional interpolation weights
            float s1 = x - i0;
            float t1 = y - j0;
            float r1 = z - k0;

            // 4. Trilinear interpolation of each velocity component

            // --- U component (X velocity) ---
            float u_x0 = Hlsl.Lerp(u[i0, j0, k0], u[i1, j0, k0], s1);
            float u_x1 = Hlsl.Lerp(u[i0, j1, k0], u[i1, j1, k0], s1);
            float u_x2 = Hlsl.Lerp(u[i0, j0, k1], u[i1, j0, k1], s1);
            float u_x3 = Hlsl.Lerp(u[i0, j1, k1], u[i1, j1, k1], s1);
            float u_y0 = Hlsl.Lerp(u_x0, u_x1, t1);
            float u_y1 = Hlsl.Lerp(u_x2, u_x3, t1);
            float finalU = Hlsl.Lerp(u_y0, u_y1, r1);

            // --- V component (Y velocity) ---
            float v_x0 = Hlsl.Lerp(v[i0, j0, k0], v[i1, j0, k0], s1);
            float v_x1 = Hlsl.Lerp(v[i0, j1, k0], v[i1, j1, k0], s1);
            float v_x2 = Hlsl.Lerp(v[i0, j0, k1], v[i1, j0, k1], s1);
            float v_x3 = Hlsl.Lerp(v[i0, j1, k1], v[i1, j1, k1], s1);
            float v_y0 = Hlsl.Lerp(v_x0, v_x1, t1);
            float v_y1 = Hlsl.Lerp(v_x2, v_x3, t1);
            float finalV = Hlsl.Lerp(v_y0, v_y1, r1);

            // --- W component (Z velocity) ---
            float w_x0 = Hlsl.Lerp(w[i0, j0, k0], w[i1, j0, k0], s1);
            float w_x1 = Hlsl.Lerp(w[i0, j1, k0], w[i1, j1, k0], s1);
            float w_x2 = Hlsl.Lerp(w[i0, j0, k1], w[i1, j0, k1], s1);
            float w_x3 = Hlsl.Lerp(w[i0, j1, k1], w[i1, j1, k1], s1);
            float w_y0 = Hlsl.Lerp(w_x0, w_x1, t1);
            float w_y1 = Hlsl.Lerp(w_x2, w_x3, t1);
            float finalW = Hlsl.Lerp(w_y0, w_y1, r1);

            // 5. The particle takes the fluid velocity, then moves with drag and damping
            //    (same integration as ParticleMoveShader with a zero acceleration)
            float3 velocity = new float3(finalU, finalV, finalW);
            velocity -= velocity * drag * timeStep;
            velocity *= damping;

            velocities[i] = velocity;
            positions[i] = pos + velocity * timeStep;
        }
    }

    /// <summary>
    /// Sums the spring tension acting on each particle. A particle only visits
    /// its own springs and only writes its own acceleration, so the pass is race free.
    /// </summary>
    [AutoConstructor]
    [EmbeddedBytecode(DispatchAxis.X)]
    public readonly partial struct SpringForcesShader : IComputeShader
    {
        private readonly ReadWriteBuffer<float3> positions;
        private readonly ReadWriteBuffer<float3> accelerations;
        private readonly ReadOnlyBuffer<int2> adjacencyRanges;
        private readonly ReadOnlyBuffer<int> adjacencyNeighbours;
        private readonly ReadOnlyBuffer<float> adjacencyRestLengths;
        private readonly float k, mass;

        public void Execute()
        {
            int particleId = ThreadIds.X;
            int2 range = adjacencyRanges[particleId];
            float3 pos = positions[particleId];
            float3 totalForce = float3.Zero;

            for (int n = range.X; n < range.Y; n++)
            {
                // Vector from this particle to the other end of the spring
                float3 forceVec = positions[adjacencyNeighbours[n]] - pos;
                float currentLength = Hlsl.Length(forceVec);

                if (currentLength > 0.001f)
                {
                    // Hooke's law: pulled towards the neighbour when stretched, pushed away when compressed
                    totalForce += (forceVec / currentLength) * (k * (currentLength - adjacencyRestLengths[n]));
                }
            }

            accelerations[particleId] = totalForce / mass;
        }
    }

    /// <summary>Integrates the acceleration into velocity and position.</summary>
    [AutoConstructor]
    [EmbeddedBytecode(DispatchAxis.X)]
    public readonly partial struct ParticleMoveShader : IComputeShader
    {
        private readonly ReadWriteBuffer<float3> positions;
        private readonly ReadWriteBuffer<float3> velocities;
        private readonly ReadWriteBuffer<float3> accelerations;
        private readonly float timeStep, damping, drag;

        public void Execute()
        {
            int i = ThreadIds.X;
            float3 velocity = velocities[i];
            float3 acc = accelerations[i] - velocity * drag;

            velocity += acc * timeStep;
            velocity *= damping;

            velocities[i] = velocity;
            positions[i] += velocity * timeStep;
        }
    }
}
