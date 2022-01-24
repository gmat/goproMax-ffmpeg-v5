/*
 * Copyright (c) 2021 Ronan LE MEILLAT
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "framesync.h"
#include "internal.h"
#include "opencl.h"
#include "opencl_source.h"
#include "video.h"

#define _WIDTH 5376
#define _HEIGHT 2688
#define OVERLAP 64
#define CUT 688
#define BASESIZE 4096 //OVERLAP and CUT are based on this size

typedef struct GoProMaxOpenCLContext {
    OpenCLFilterContext ocf;

    int              initialised;
    cl_kernel        kernel;
    cl_command_queue command_queue;

    FFFrameSync      fs;

    int              nb_planes;
    int              x_subsample;
    int              y_subsample;
    int              alpha_separate;

    int              eac_output;
} GoProMaxOpenCLContext;

static int gopromax_opencl_load(AVFilterContext *avctx,
                               enum AVPixelFormat gopromax_front_format,
                               enum AVPixelFormat gopromax_rear_format)
{
    GoProMaxOpenCLContext *ctx = avctx->priv;
    cl_int cle;
    const char *source = ff_opencl_source_gopromax;
    const char *kernel;
    const AVPixFmtDescriptor *gopromax_front_desc, *gopromax_rear_desc;
    int err, i, gopromax_front_planes, gopromax_rear_planes;

    gopromax_front_desc    = av_pix_fmt_desc_get(gopromax_front_format);
    gopromax_rear_desc = av_pix_fmt_desc_get(gopromax_rear_format);
    gopromax_front_planes = gopromax_rear_planes = 0;
    for (i = 0; i < gopromax_front_desc->nb_components; i++)
        gopromax_front_planes = FFMAX(gopromax_front_planes,
                            gopromax_front_desc->comp[i].plane + 1);
    for (i = 0; i < gopromax_rear_desc->nb_components; i++)
        gopromax_rear_planes = FFMAX(gopromax_rear_planes,
                               gopromax_rear_desc->comp[i].plane + 1);

    ctx->nb_planes = gopromax_front_planes;
    ctx->x_subsample = 1 << gopromax_front_desc->log2_chroma_w;
    ctx->y_subsample = 1 << gopromax_front_desc->log2_chroma_h;

    
    if (ctx->eac_output >0 )
            {
                kernel = "gopromax_stack";
            }
    else {
        kernel = "gopromax_equirectangular";
    }

    av_log(avctx, AV_LOG_DEBUG, "Using kernel %s.\n", kernel);

    err = ff_opencl_filter_load_program(avctx, &source, 1);
    av_log(avctx, AV_LOG_VERBOSE,"OpenCL Kernel %s loaded err=%d\n",kernel,err);
    if (err < 0)
        goto fail;

    ctx->command_queue = clCreateCommandQueue(ctx->ocf.hwctx->context,
                                              ctx->ocf.hwctx->device_id,
                                              0, &cle);
    av_log(avctx, AV_LOG_VERBOSE,"Leaving loading\n");
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create OpenCL "
                     "command queue %d.\n", cle);

    ctx->kernel = clCreateKernel(ctx->ocf.program, kernel, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel %d.\n", cle);

    ctx->initialised = 1;

    return 0;

fail:
    if (ctx->command_queue)
        clReleaseCommandQueue(ctx->command_queue);
    if (ctx->kernel)
        clReleaseKernel(ctx->kernel);
    return err;
}

static int gopromax_opencl_stack(FFFrameSync *fs)
{
    AVFilterContext    *avctx = fs->parent;
    AVFilterLink     *outlink = avctx->outputs[0];
    
    GoProMaxOpenCLContext *ctx = avctx->priv;
    AVFrame *gopromax_front, *gotpromax_rear;
    AVFrame *output;
    cl_mem mem;
    cl_int cle;//, x, y;
    size_t global_work[2];
    int kernel_arg = 0;
    int err, plane;

    err = ff_framesync_get_frame(fs, 0, &gopromax_front, 0);
    if (err < 0)
        return err;
    err = ff_framesync_get_frame(fs, 1, &gotpromax_rear, 0);
    if (err < 0)
        return err;
    
    if (!ctx->initialised) {
        AVHWFramesContext *gopromax_front_fc =
            (AVHWFramesContext*)gopromax_front->hw_frames_ctx->data;
        AVHWFramesContext *gopromax_rear_fc =
            (AVHWFramesContext*)gotpromax_rear->hw_frames_ctx->data;
        err = gopromax_opencl_load(avctx, gopromax_front_fc->sw_format,
                                  gopromax_rear_fc->sw_format);
        if (err < 0)
            return err;
    }

    output = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!output) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    for (plane = 0; plane < ctx->nb_planes; plane++) {
        kernel_arg = 0;

        mem = (cl_mem)output->data[plane];
        CL_SET_KERNEL_ARG(ctx->kernel, kernel_arg, cl_mem, &mem);
        kernel_arg++;

        mem = (cl_mem)gopromax_front->data[plane];
        CL_SET_KERNEL_ARG(ctx->kernel, kernel_arg, cl_mem, &mem);
        kernel_arg++;

        mem = (cl_mem)gotpromax_rear->data[plane];
        CL_SET_KERNEL_ARG(ctx->kernel, kernel_arg, cl_mem, &mem);
        kernel_arg++;

        err = ff_opencl_filter_work_size_from_image(avctx, global_work,
                                                    output, plane, 0);
        if (err < 0)
            goto fail;

        av_log(avctx, AV_LOG_VERBOSE,"In gopromax_opencl_stack for plane:%d %dx%d frame size %dx%d\n",plane,global_work[0],global_work[1],outlink->w, outlink->h);
        
        cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->kernel, 2, NULL,
                                     global_work, NULL, 0, NULL, NULL);
        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue gopromax kernel "
                         "for plane %d: %d.\n", plane, cle);
    }

    cle = clFinish(ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue: %d.\n", cle);

    err = av_frame_copy_props(output, gopromax_front);

    av_log(avctx, AV_LOG_DEBUG, "Filter output: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(output->format),
           output->width, output->height, output->pts);

    return ff_filter_frame(outlink, output);

fail:
    av_frame_free(&output);
    return err;
}

static int gopromax_opencl_config_output(AVFilterLink *outlink)
{
    AVFilterContext *avctx = outlink->src;
    av_log(avctx, AV_LOG_VERBOSE,"Setting output\n");
    GoProMaxOpenCLContext *ctx = avctx->priv;
    av_log(avctx, AV_LOG_VERBOSE,"Geting filtercontext\n");
    AVFilterLink *inlink = avctx->inputs[0];
    const AVPixFmtDescriptor *desc_in  = av_pix_fmt_desc_get(inlink->format);
 
    int height = avctx->inputs[0]->h;
    int width = avctx->inputs[0]->w;
    int err;
    
    if (desc_in->log2_chroma_w != desc_in->log2_chroma_h) {
        av_log(avctx, AV_LOG_ERROR, "Input format %s not supported.\n",
               desc_in->name);
        return AVERROR(EINVAL);
    }
    
    if (ctx->eac_output==0)
    {
        ctx->ocf.output_width = 4*height;
        ctx->ocf.output_height = 2*height;
    }
    else
    {
        int overlap = width *  OVERLAP / BASESIZE;
        ctx->ocf.output_width = width - 2*overlap;
        ctx->ocf.output_height = 2*height;
    }

    err = ff_opencl_filter_config_output(outlink);
    av_log(avctx, AV_LOG_VERBOSE,"Output config ok w=%d h=%d err=%d\n",outlink->w, outlink->h, err);
    if (err < 0)
        return err;

    err = ff_framesync_init_dualinput(&ctx->fs, avctx);
    av_log(avctx, AV_LOG_VERBOSE,"Dualinput config ok err=%d\n",err);
    if (err < 0)
        return err;

    return ff_framesync_configure(&ctx->fs);
}

static av_cold int gopromax_opencl_init(AVFilterContext *avctx)
{
    GoProMaxOpenCLContext *ctx = avctx->priv;

    ctx->fs.on_event = &gopromax_opencl_stack;

    return ff_opencl_filter_init(avctx);
}

static int gopromax_opencl_activate(AVFilterContext *avctx)
{
    GoProMaxOpenCLContext *ctx = avctx->priv;

    return ff_framesync_activate(&ctx->fs);
}

static av_cold void gopromax_opencl_uninit(AVFilterContext *avctx)
{
    GoProMaxOpenCLContext *ctx = avctx->priv;
    cl_int cle;

    if (ctx->kernel) {
        cle = clReleaseKernel(ctx->kernel);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "kernel: %d.\n", cle);
    }

    if (ctx->command_queue) {
        cle = clReleaseCommandQueue(ctx->command_queue);
        if (cle != CL_SUCCESS)
            av_log(avctx, AV_LOG_ERROR, "Failed to release "
                   "command queue: %d.\n", cle);
    }

    ff_opencl_filter_uninit(avctx);

    ff_framesync_uninit(&ctx->fs);
}

#define OFFSET(x) offsetof(GoProMaxOpenCLContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption gopromax_opencl_options[] = {
    { "eac", "output Equiangular cubemap",
      OFFSET(eac_output), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, .flags = FLAGS },
    { NULL },
};

AVFILTER_DEFINE_CLASS(gopromax_opencl);

static const AVFilterPad gopromax_opencl_inputs[] = {
    {
        .name         = "gopromax_front",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_opencl_filter_config_input,
    },
    {
        .name         = "gopromax_rear",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_opencl_filter_config_input,
    },
};

static const AVFilterPad gopromax_opencl_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = &gopromax_opencl_config_output,
    },
};

const AVFilter ff_vf_gopromax_opencl = {
    .name            = "gopromax_opencl",
    .description     = NULL_IF_CONFIG_SMALL("GoProMax .360 to equirectangular projection"),
    .priv_size       = sizeof(GoProMaxOpenCLContext),
    .priv_class      = &gopromax_opencl_class,
    .init            = &gopromax_opencl_init,
    .uninit          = &gopromax_opencl_uninit,
    .activate        = &gopromax_opencl_activate,
    FILTER_INPUTS(gopromax_opencl_inputs),
    FILTER_OUTPUTS(gopromax_opencl_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_OPENCL),
    .flags_internal  = FF_FILTER_FLAG_HWFRAME_AWARE,
};
