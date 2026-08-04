/* lcevc_verify — decode an LCEVC enhancement stream (plus base pictures)
 * with the reference decoder (liblcevc_dec) and compare the output with the
 * encoder's own reconstruction.
 *
 * Usage:
 *   lcevc_verify <enhancement.lcevc> <base.yuv> <recon.yuv> <baseW> <baseH> <W> <H>
 *
 * The enhancement file must contain one NAL unit per frame, concatenated.
 * base.yuv and recon.yuv are yuv420p frame dumps (one frame per encoded
 * frame), as produced by `lcevc_enc --dump-base --dump-recon`.
 */

#include <LCEVC/lcevc_dec.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint64_t read_u64_le(const uint8_t *p)
{
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

static int is_start_code(const uint8_t *p)
{
    return p[0] == 0 && p[1] == 0 && p[2] == 1;
}

/* Read the next NAL unit (0x000001 start code framed) from the file.
 * Returns the unit including its start code. */
static int read_nal(FILE *f, uint8_t **out, size_t *out_size)
{
    uint8_t *buf = NULL;
    size_t cap = 0;
    size_t len = 0;
    int first = 1;

    /* Skip the leading start code (00 00 01, optionally 00 00 00 01). */
    int b0 = fgetc(f);
    if (b0 == EOF)
        return 0;
    int b1 = fgetc(f);
    int b2 = fgetc(f);
    if (b0 != 0 || b1 != 0 || b2 != 1) {
        if (b0 == 0 && b1 == 0 && b2 == 0) {
            int b3 = fgetc(f);
            if (b3 != 1) {
                fprintf(stderr, "b3!=1: %02x\n", b3);
                free(buf);
                return -1;
            }
        } else {
            fprintf(stderr, "bad start: %02x %02x %02x\n", b0, b1, b2);
            free(buf);
            return -1;
        }
    }
    /* Emit the start code as part of the unit. */
    uint8_t sc[4] = {0, 0, 0, 1};
    size_t sc_len = (b0 == 0 && b1 == 0 && b2 == 0) ? 4 : 3;
    buf = malloc(sc_len);
    memcpy(buf, sc + (4 - sc_len), sc_len);
    len = sc_len;
    cap = sc_len;

    for (;;) {
        int c = fgetc(f);
        if (c == EOF)
            break;
        first = 0;

        if (len > sc_len && len >= 3 && is_start_code(buf + len - 3)) {
            /* Previous NAL ends before this start code. The start code sits
             * at file positions (len - 3)..(len - 1); the read pointer is at
             * len + 1, so rewind 4 to land on the start code. */
            fseek(f, -4, SEEK_CUR);
            len -= 3;
            break;
        }

        if (len + 1 > cap) {
            cap = cap ? cap * 2 : 65536;
            uint8_t *nb = realloc(buf, cap);
            if (!nb) { fprintf(stderr, "realloc fail\n"); free(buf); return -1; }
            buf = nb;
        }
        buf[len++] = (uint8_t)c;
    }
    (void)first;

    *out = buf;
    *out_size = len;
    return 1;
}

static int write_plane(uint8_t *dst, const uint8_t *src, int w, int h, int stride)
{
    for (int y = 0; y < h; y++)
        memcpy(dst + (size_t)y * w, src + (size_t)y * stride, w);
    return 0;
}

static void on_event(LCEVC_DecoderHandle dec, LCEVC_Event event, LCEVC_PictureHandle pic,
                     const LCEVC_DecodeInformation *info, const uint8_t *data, uint32_t dataSize,
                     void *userData)
{
    (void)dec;
    (void)pic;
    (void)info;
    (void)userData;
    if (event == LCEVC_Log && data && dataSize) {
        fwrite(data, 1, dataSize, stderr);
        fputc('\n', stderr);
    }
}

int main(int argc, char **argv)
{
    if (argc != 8) {
        fprintf(stderr, "usage: %s <enh.lcevc> <base.yuv> <recon.yuv> <baseW> <baseH> <W> <H>\n", argv[0]);
        return 2;
    }
    const char *enh_path = argv[1];
    const char *base_path = argv[2];
    const char *recon_path = argv[3];
    uint32_t baseW = (uint32_t)atoi(argv[4]);
    uint32_t baseH = (uint32_t)atoi(argv[5]);
    uint32_t W = (uint32_t)atoi(argv[6]);
    uint32_t H = (uint32_t)atoi(argv[7]);

    FILE *enh = fopen(enh_path, "rb");
    FILE *basef = fopen(base_path, "rb");
    FILE *reconf = fopen(recon_path, "rb");
    if (!enh || !basef || !reconf) {
        fprintf(stderr, "failed to open inputs\n");
        return 2;
    }

    size_t base_frame_size = (size_t)baseW * baseH * 3 / 2;
    size_t out_frame_size = (size_t)W * H * 3 / 2;
    uint8_t *base_frame = malloc(base_frame_size);
    uint8_t *recon_frame = malloc(out_frame_size);
    if (!base_frame || !recon_frame) return 2;

    LCEVC_DecoderHandle decoder = {0};
    LCEVC_ReturnCode rc = LCEVC_CreateDecoder(&decoder, (LCEVC_AccelContextHandle){0});
    if (rc != LCEVC_Success) { fprintf(stderr, "CreateDecoder failed: %d\n", rc); return 1; }
    int32_t events[] = {LCEVC_Log, LCEVC_Exit, LCEVC_OutputPictureDone, LCEVC_BasePictureDone,
                        LCEVC_CanSendBase, LCEVC_CanSendEnhancement, LCEVC_CanSendPicture,
                        LCEVC_CanReceive};
    LCEVC_ConfigureDecoderIntArray(decoder, "events", 8, events);
    LCEVC_SetDecoderEventCallback(decoder, on_event, NULL);
    rc = LCEVC_InitializeDecoder(decoder);
    if (rc != LCEVC_Success) { fprintf(stderr, "InitializeDecoder failed: %d\n", rc); return 1; }

    LCEVC_PictureDesc baseDesc;
    LCEVC_DefaultPictureDesc(&baseDesc, LCEVC_I420_8, baseW, baseH);
    LCEVC_PictureDesc outDesc;
    LCEVC_DefaultPictureDesc(&outDesc, LCEVC_I420_8, W, H);

    uint64_t frame = 0;
    uint64_t max_diff_all = 0;
    uint32_t fx = 0, fy = 0, fp = 0; int fdv = 0, fev = 0;
    long long sse_all = 0;
    long long n_all = 0;

    for (;;) {
        uint8_t *nal = NULL;
        size_t nal_size = 0;
        int r = read_nal(enh, &nal, &nal_size);
        if (r == 0) break;
        if (r < 0) { fprintf(stderr, "read_nal failed\n"); return 1; }

        if (fread(base_frame, 1, base_frame_size, basef) != base_frame_size) {
            fprintf(stderr, "base file ended early\n");
            return 1;
        }
        if (fread(recon_frame, 1, out_frame_size, reconf) != out_frame_size) {
            fprintf(stderr, "recon file ended early\n");
            return 1;
        }

        /* Create and fill the base picture. */
        LCEVC_PictureHandle basePic = {0};
        LCEVC_AllocPicture(decoder, &baseDesc, &basePic);
        LCEVC_PictureLockHandle lock;
        if (LCEVC_LockPicture(decoder, basePic, LCEVC_Access_Write, &lock) == LCEVC_Success) {
            uint32_t planes = 0;
            LCEVC_GetPicturePlaneCount(decoder, basePic, &planes);
            size_t off = 0;
            for (uint32_t p = 0; p < planes; p++) {
                LCEVC_PicturePlaneDesc pd = {0};
                LCEVC_GetPictureLockPlaneDesc(decoder, lock, p, &pd);
                uint32_t pw = baseW / (p ? 2 : 1);
                uint32_t ph = baseH / (p ? 2 : 1);
                write_plane((uint8_t *)pd.firstSample, base_frame + off, pw, ph, pd.rowByteStride);
                off += (size_t)pw * ph;
            }
            LCEVC_UnlockPicture(decoder, lock);
        }

        /* Output picture. */
        LCEVC_PictureHandle outPic = {0};
        LCEVC_AllocPicture(decoder, &outDesc, &outPic);

        fprintf(stderr, "nal size: %zu first bytes: %02x %02x %02x %02x %02x\n", nal_size, nal[0], nal[1], nal[2], nal[3], nal[4]);
        LCEVC_SendDecoderEnhancementData(decoder, frame, nal, (uint32_t)nal_size);
        LCEVC_SendDecoderBase(decoder, frame, basePic, 1000000, NULL);
        LCEVC_SendDecoderPicture(decoder, outPic);

        LCEVC_PictureHandle decoded = {0};
        LCEVC_DecodeInformation info = {0};
        int tries = 0;
        while ((rc = LCEVC_ReceiveDecoderPicture(decoder, &decoded, &info)) == LCEVC_Again && tries++ < 1000) {
            LCEVC_SynchronizeDecoder(decoder, false);
        }
        if (rc != LCEVC_Success) {
            fprintf(stderr, "frame %llu: ReceiveDecoderPicture failed: %d\n",
                    (unsigned long long)frame, rc);
            return 1;
        }

        /* Compare with the encoder reconstruction. */
        if (LCEVC_LockPicture(decoder, decoded, LCEVC_Access_Read, &lock) == LCEVC_Success) {
            uint32_t planes = 0;
            LCEVC_GetPicturePlaneCount(decoder, decoded, &planes);
            size_t off = 0;
            for (uint32_t p = 0; p < planes; p++) {
                LCEVC_PicturePlaneDesc pd = {0};
                rc = LCEVC_GetPictureLockPlaneDesc(decoder, lock, p, &pd);
                if (rc != LCEVC_Success || !pd.firstSample) {
                    fprintf(stderr, "frame %llu: plane %u desc failed: %d\n",
                            (unsigned long long)frame, p, rc);
                    return 1;
                }
                fprintf(stderr, "  plane %u: firstSample=%p stride=%d\n", p,
                        pd.firstSample, pd.rowByteStride);
                uint32_t pw = W / (p ? 2 : 1);
                uint32_t ph = H / (p ? 2 : 1);
                for (uint32_t y = 0; y < ph; y++) {
                    const uint8_t *src = (const uint8_t *)pd.firstSample + (size_t)y * pd.rowByteStride;
                    char dname[64];
                    snprintf(dname, sizeof(dname), "/tmp/opencode/dec_f%llu.yuv", (unsigned long long)frame);
                    FILE *dump = fopen(dname, "ab");
                    fwrite(src, 1, pw, dump);
                    fclose(dump);
                    for (uint32_t x = 0; x < pw; x++) {
                        int a = src[x];
                        int b = recon_frame[off + y * pw + x];
                        int d = a > b ? a - b : b - a;
                        if ((uint64_t)d > max_diff_all) { if (max_diff_all == 0) fprintf(stderr, "first diff at %u,%u p%d: dec=%d enc=%d\n", x, y, p, a, b); max_diff_all = d; fx = x; fy = y; fp = p; fdv = a; fev = b; }
                        if (frame == 0 && y < 8 && x < 8) { fprintf(stderr, "px %u,%u: dec=%d enc=%d ", x, y, a, b); }
                        sse_all += (long long)d * d;
                        n_all++;
                    }
                }
                off += (size_t)pw * ph;
            }
            LCEVC_UnlockPicture(decoder, lock);
        } else {
            fprintf(stderr, "frame %llu: LockPicture failed\n", (unsigned long long)frame);
            return 1;
        }

        LCEVC_FreePicture(decoder, decoded);
        LCEVC_PictureHandle doneBase = {0};
        if (LCEVC_ReceiveDecoderBase(decoder, &doneBase) == LCEVC_Success)
            LCEVC_FreePicture(decoder, doneBase);

        double psnr = n_all ? 10.0 * log10((double)255 * 255 * n_all / (double)sse_all) : 999.0;
        printf("frame %llu: decoded ok, PSNR %.3f dB, max diff %llu at %u,%u p%u (dec=%d enc=%d)\n",
               (unsigned long long)frame, psnr, (unsigned long long)max_diff_all, fx, fy, fp, fdv, fev);
        fflush(stdout);

        free(nal);
        frame++;
    }

    double psnr = n_all ? 10.0 * log10((double)255 * 255 * n_all / (double)sse_all) : 999.0;
    printf("RESULT: %llu frames decoded, PSNR vs encoder reconstruction: %.3f dB, max diff: %llu (at %u,%u p%u dec=%d enc=%d)\n",
           (unsigned long long)frame, psnr, (unsigned long long)max_diff_all, fx, fy, fp, fdv, fev);

    LCEVC_DestroyDecoder(decoder);
    fclose(enh);
    fclose(basef);
    fclose(reconf);
    return 0;
}
