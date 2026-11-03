// Hand-written companion to the UniFFI-generated bindings (bindings/kotlin), shipped in the
// same package. It exists for speed: UniFFI's Kotlin bindings pass every record through JNA
// structures, which costs about a millisecond per call on Android, more than preparing a frame
// takes. This calls the core's plain C entry point `vitallens_prepare_frame` through JNA direct
// mapping instead, with only numbers and addresses as arguments.

package com.rouast.vitallens.core

import com.sun.jna.Native
import com.sun.jna.Pointer
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * One plane of a frame in a direct [ByteBuffer], starting at the buffer's position and spanning
 * its remaining bytes, as in `android.media.Image.Plane` or CameraX `ImageProxy.PlaneProxy`.
 */
class FramePlaneBuffer(val buffer: ByteBuffer, val rowStride: Int, val pixelStride: Int)

/**
 * Crops [crop] out of a frame, converts it to RGB, area-averages it to [outWidth] x [outHeight]
 * (the size after rotation), then rotates and mirrors it per [orientation], in one native pass.
 * Returns packed RGB bytes (`outWidth * outHeight * 3`, row-major).
 *
 * The planes are read in place, without copying; they must stay valid until this returns (e.g.
 * keep the `Image` or `ImageProxy` open). Packed formats take one plane, [PixelFormat.YUV420]
 * takes Y, U and V. [yuvMatrix] applies to YUV only.
 *
 * Thread-safe; each thread reuses its own small output buffer.
 */
@Throws(FrameException::class)
fun prepareFrame(
    planes: List<FramePlaneBuffer>,
    width: Int,
    height: Int,
    format: PixelFormat,
    yuvMatrix: YuvMatrix,
    crop: PixelRect,
    outWidth: Int,
    outHeight: Int,
    orientation: Orientation,
): ByteArray = NativeFrames.prepare(planes, width, height, format, yuvMatrix, crop, outWidth, outHeight, orientation)

private object NativeFrames {
    init {
        Native.register(
            NativeFrames::class.java,
            System.getProperty("uniffi.component.vitallens_core.libraryOverride") ?: "vitallens_core",
        )
    }

    @JvmStatic
    external fun vitallens_prepare_frame(
        p0: Long, p0Len: Long, p0RowStride: Int, p0PixelStride: Int,
        p1: Long, p1Len: Long, p1RowStride: Int, p1PixelStride: Int,
        p2: Long, p2Len: Long, p2RowStride: Int, p2PixelStride: Int,
        width: Int, height: Int, format: Int, yuvMatrix: Int,
        cropX: Int, cropY: Int, cropWidth: Int, cropHeight: Int,
        outWidth: Int, outHeight: Int, rotationDegrees: Int, mirrored: Int,
        out: Long, outLen: Long,
        err: Long, errCap: Int, errLen: Long,
    ): Int

    private const val ERR_CAP = 256

    private class Scratch {
        var out: ByteBuffer = ByteBuffer.allocateDirect(40 * 40 * 3)
        val err: ByteBuffer = ByteBuffer.allocateDirect(ERR_CAP + 4).order(ByteOrder.nativeOrder())
        val outAddress get() = address(out)
        val errAddress = address(err)
    }

    private val scratch = ThreadLocal.withInitial { Scratch() }

    private fun address(buffer: ByteBuffer): Long {
        require(buffer.isDirect) { "frame planes must be direct ByteBuffers" }
        return Pointer.nativeValue(Native.getDirectBufferPointer(buffer)) + buffer.position()
    }

    private fun formatCode(f: PixelFormat) = when (f) {
        PixelFormat.RGB -> 0
        PixelFormat.BGR -> 1
        PixelFormat.RGBA -> 2
        PixelFormat.BGRA -> 3
        PixelFormat.ARGB -> 4
        PixelFormat.YUV420 -> 5
    }

    private fun matrixCode(m: YuvMatrix) = when (m) {
        YuvMatrix.BT601_LIMITED -> 0
        YuvMatrix.BT601_FULL -> 1
        YuvMatrix.BT709_LIMITED -> 2
        YuvMatrix.BT709_FULL -> 3
    }

    fun prepare(
        planes: List<FramePlaneBuffer>, width: Int, height: Int, format: PixelFormat, yuvMatrix: YuvMatrix,
        crop: PixelRect, outWidth: Int, outHeight: Int, orientation: Orientation,
    ): ByteArray {
        val expected = if (format == PixelFormat.YUV420) 3 else 1
        if (planes.size != expected) throw FrameException.InvalidInput("$format needs $expected plane(s), got ${planes.size}")
        if (outWidth <= 0 || outHeight <= 0) throw FrameException.InvalidInput("output size must be positive")
        val s = scratch.get()!!
        val outLen = outWidth * outHeight * 3
        if (s.out.capacity() < outLen) s.out = ByteBuffer.allocateDirect(outLen)
        val addr = LongArray(3)
        val len = LongArray(3)
        val rs = IntArray(3)
        val ps = IntArray(3)
        for ((i, p) in planes.withIndex()) {
            addr[i] = address(p.buffer)
            len[i] = p.buffer.remaining().toLong()
            rs[i] = p.rowStride
            ps[i] = p.pixelStride
        }
        val code = vitallens_prepare_frame(
            addr[0], len[0], rs[0], ps[0],
            addr[1], len[1], rs[1], ps[1],
            addr[2], len[2], rs[2], ps[2],
            width, height, formatCode(format), matrixCode(yuvMatrix),
            crop.x.toInt(), crop.y.toInt(), crop.width.toInt(), crop.height.toInt(),
            outWidth, outHeight, orientation.rotationDegrees.toInt(), if (orientation.mirrored) 1 else 0,
            s.outAddress, outLen.toLong(),
            s.errAddress + 4, ERR_CAP, s.errAddress,
        )
        if (code != 0) {
            val n = s.err.getInt(0).coerceIn(0, ERR_CAP)
            val bytes = ByteArray(n)
            s.err.duplicate().apply { position(4) }.get(bytes)
            throw FrameException.InvalidInput(String(bytes, Charsets.UTF_8))
        }
        val result = ByteArray(outLen)
        s.out.duplicate().apply { position(0) }.get(result)
        return result
    }
}
