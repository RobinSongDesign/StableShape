using ComputeSharp;

namespace StableShapeGPU.Properties
{
    /// <summary>
    /// A velocity field that lives on the GPU.
    /// Consumers (e.g. the mesh solver) sample it directly inside their own shaders,
    /// so the field never has to travel through CPU memory.
    /// </summary>
    public interface IVelocityField3D
    {
        /// <summary>Grid resolution along X.</summary>
        int SizeX { get; }

        /// <summary>Grid resolution along Y.</summary>
        int SizeY { get; }

        /// <summary>Grid resolution along Z.</summary>
        int SizeZ { get; }

        /// <summary>X component of the velocity.</summary>
        ReadWriteTexture3D<float> U { get; }

        /// <summary>Y component of the velocity.</summary>
        ReadWriteTexture3D<float> V { get; }

        /// <summary>Z component of the velocity.</summary>
        ReadWriteTexture3D<float> W { get; }
    }
}
