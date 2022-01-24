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

#define OVERLAP 64
#define CUT 688
#define BASESIZE 4096 //OVERLAP and CUT are based on this size


#define FOV 360.0f
enum Faces {
    TOP_LEFT,
    TOP_MIDDLE,
    TOP_RIGHT,
    BOTTOM_LEFT,
    BOTTOM_MIDDLE,
    BOTTOM_RIGHT,
    NB_FACES,
};

enum Direction {
    RIGHT,
    LEFT,
    UP,
    DOWN,
    FRONT,
    BACK,
    NB_DIRECTIONS,
};

enum Rotation {
    ROT_0,
    ROT_90,
    ROT_180,
    ROT_270,
    NB_ROTATIONS,
};

float2 rotate_cube_face(float2 uv, int rotation);
int2 transpose_gopromax_overlap(int2 xy, int2 dim);
float3 equirect_to_xyz(int2 xy,int2 size);
float2 xyz_to_cube(float3 xyz, int *direction, int *face);
float2 xyz_to_eac(float3 xyz, int2 size);

float2 rotate_cube_face(float2 uv, int rotation)
{
    float2 ret_uv;

    switch (rotation) {
    case ROT_0:
        ret_uv = uv;
        break;
    case ROT_90:
        ret_uv.x = -uv.y;
        ret_uv.y =  uv.x;
        break;
    case ROT_180:
        ret_uv.x = -uv.x;
        ret_uv.y = -uv.y;
        break;
    case ROT_270:
        ret_uv.x =  uv.y;
        ret_uv.y =  -uv.x;
        break;
    }
return ret_uv;
}

float3 equirect_to_xyz(int2 xy,int2 size)
{
    float3 xyz;
    float phi   = ((2.f * ((float)xy.x) + 0.5f) / ((float)size.x)  - 1.f) * M_PI_F ;
    float theta = ((2.f * ((float)xy.y) + 0.5f) / ((float)size.y) - 1.f) * M_PI_2_F;

    xyz.x = cos(theta) * sin(phi);
    xyz.y = sin(theta);
    xyz.z = cos(theta) * cos(phi);

    return xyz;
}

float2 xyz_to_cube(float3 xyz, int *direction, int *face)
{
    float phi   = atan2(xyz.x, xyz.z);
    float theta = asin(xyz.y);
    float phi_norm, theta_threshold;
    int face_rotation;
    float2 uv;
    //int direction;

    if (phi >= -M_PI_4_F && phi < M_PI_4_F) {
        *direction = FRONT;
        phi_norm = phi;
    } else if (phi >= -(M_PI_2_F + M_PI_4_F) && phi < -M_PI_4_F) {
        *direction = LEFT;
        phi_norm = phi + M_PI_2_F;
    } else if (phi >= M_PI_4_F && phi < M_PI_2_F + M_PI_4_F) {
        *direction = RIGHT;
        phi_norm = phi - M_PI_2_F;
    } else {
        *direction = BACK;
        phi_norm = phi + ((phi > 0.f) ? -M_PI_F : M_PI_F);
    }

    theta_threshold = atan(cos(phi_norm));
    if (theta > theta_threshold) {
        *direction = DOWN;
    } else if (theta < -theta_threshold) {
        *direction = UP;
    }
    
    theta_threshold = atan(cos(phi_norm));
    if (theta > theta_threshold) {
        *direction = DOWN;
    } else if (theta < -theta_threshold) {
        *direction = UP;
    }

    switch (*direction) {
    case RIGHT:
        uv.x = -xyz.z / xyz.x;
        uv.y =  xyz.y / xyz.x;
        *face = TOP_RIGHT;
        face_rotation = ROT_0;
        break;
    case LEFT:
        uv.x = -xyz.z / xyz.x;
        uv.y = -xyz.y / xyz.x;
        *face = TOP_LEFT;
        face_rotation = ROT_0;
        break;
    case UP:
        uv.x = -xyz.x / xyz.y;
        uv.y = -xyz.z / xyz.y;
        *face = BOTTOM_RIGHT;
        face_rotation = ROT_270;
        uv = rotate_cube_face(uv,face_rotation);
        break;
    case DOWN:
        uv.x =  xyz.x / xyz.y;
        uv.y = -xyz.z / xyz.y;
        *face = BOTTOM_LEFT;
        face_rotation = ROT_270;
        uv = rotate_cube_face(uv,face_rotation);
        break;
    case FRONT:
        uv.x =  xyz.x / xyz.z;
        uv.y =  xyz.y / xyz.z;
        *face = TOP_MIDDLE;
        face_rotation = ROT_0;
        break;
    case BACK:
        uv.x =  xyz.x / xyz.z;
        uv.y = -xyz.y / xyz.z;
        *face = BOTTOM_MIDDLE;
        face_rotation = ROT_90;
        uv = rotate_cube_face(uv,face_rotation);
        break;
    }
    
    return uv;
}

float2 xyz_to_eac(float3 xyz, int2 size)
{
    float pixel_pad = 2;
    float u_pad = pixel_pad / size.x;
    float v_pad = pixel_pad / size.y;

    int direction, face;
    int u_face, v_face;
    float2 uv = xyz_to_cube(xyz,&direction,&face);

    u_face = face % 3;
    v_face = face / 3;
    //eac expansion
    uv.x = M_2_PI_F * atan(uv.x) + 0.5f;
    uv.y = M_2_PI_F * atan(uv.y) + 0.5f;
    
    uv.x = (uv.x + u_face) * (1.f - 2.f * u_pad) / 3.f + u_pad;
    uv.y = uv.y * (0.5f - 2.f * v_pad) + v_pad + 0.5f * v_face;
    
    uv.x *= size.x;
    uv.y *= size.y;

    return uv;
}

const sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_CLAMP_TO_EDGE   |
                           CLK_FILTER_NEAREST);

int2 transpose_gopromax_overlap(int2 xy, int2 dim)
{
    int2 ret;
    int cut = dim.x*CUT/BASESIZE;
    int overlap = dim.x*OVERLAP/BASESIZE;
    if (xy.x<cut)
        {
            ret = xy;
        }
    else if ((xy.x>=cut) && (xy.x< (dim.x-cut)))
        {
            ret.x = xy.x+overlap;
            ret.y = xy.y;
        }
    else
        {
            ret.x = xy.x+2*overlap;
            ret.y = xy.y;
        }
    return ret;
}
__kernel void gopromax_equirectangular(__write_only image2d_t dst,
                             __read_only  image2d_t gopromax_front,
                             __read_only  image2d_t gopromax_rear)
{
    
    float4 val;
    int2 loc = (int2)(get_global_id(0), get_global_id(1));

    int2 dst_size = get_image_dim(dst);
    int2 src_size = get_image_dim(gopromax_front);
    int2 eac_size = (int2)(src_size.x-2*(src_size.x*OVERLAP/BASESIZE),dst_size.y);

    int half_eight = src_size.y;
    
    float3 xyz = equirect_to_xyz(loc,dst_size);

    float2 uv = xyz_to_eac(xyz,eac_size);
    
    int2 xy = convert_int2(floor(uv));

    xy = transpose_gopromax_overlap(xy,eac_size);
    
    if (xy.y<half_eight)
        {
            val = read_imagef(gopromax_front,sampler,xy);
        }
    else
        {
            val = read_imagef(gopromax_rear,sampler,(int2)(xy.x, (xy.y-half_eight)));
        }

    write_imagef(dst, loc, val);

}

__kernel void gopromax_stack(__write_only image2d_t dst,
                             __read_only  image2d_t gopromax_front,
                             __read_only  image2d_t gopromax_rear)
{
    const sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE   |
                               CLK_FILTER_NEAREST);
    
    float4 val;
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    int2 dst_size = get_image_dim(dst);
    int half_height = dst_size.y / 2;
    int cut0 = dst_size.x * CUT / (BASESIZE-2*OVERLAP);
    int cut1 = dst_size.x - cut0;
    int overlap = dst_size.x * OVERLAP / (BASESIZE-2*OVERLAP);
    
    int x;
    if (loc.x < (cut0-overlap))
    {
        x = loc.x;
    }
    else if ( (loc.x>=(cut0-overlap)) && ( loc.x < ( cut1 + overlap) ) )
    {
        x = loc.x + overlap;
    }
    else if ( loc.x >= ( cut1 - 2*overlap) )
    {
        x = loc.x + 2*overlap;
    }
    
    if (loc.y < half_height)
    {
        val = read_imagef(gopromax_front, sampler, (int2)(x, loc.y));
    }
    else
    {
        val = read_imagef(gopromax_rear, sampler, (int2)(x, loc.y-half_height));
    }

        write_imagef(dst, loc, val);
}
