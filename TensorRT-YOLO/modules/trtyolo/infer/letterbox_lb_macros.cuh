#pragma once

// Assumes the following symbols exist in the including TU scope:
// - grid, block, stream, MINB
// - src, src_cols, src_rows, src_pitch
// - dst, dst_cols, dst_rows
// - params, config
// - (optional) num_images for multi-image variant

// Compact MINB dispatch (single image)
#define LB_MINB_CALL_SINGLE(KERNEL,BX,BY) do{ \
 if(MINB<=1)      KERNEL<BX,BY,1><<<grid,block,0,stream>>>(src,src_cols,src_rows,src_pitch,dst,dst_cols,dst_rows,params,config); \
 else if(MINB==2) KERNEL<BX,BY,2><<<grid,block,0,stream>>>(src,src_cols,src_rows,src_pitch,dst,dst_cols,dst_rows,params,config); \
 else             KERNEL<BX,BY,3><<<grid,block,0,stream>>>(src,src_cols,src_rows,src_pitch,dst,dst_cols,dst_rows,params,config); \
}while(0)

// Compact MINB dispatch (multi image)
#define LB_MINB_CALL_MULTI(KERNEL,BX,BY) do{ \
 if(MINB<=1)      KERNEL<BX,BY,1><<<grid,block,0,stream>>>(src,src_cols,src_rows,src_pitch,dst,dst_cols,dst_rows,params,config,num_images); \
 else if(MINB==2) KERNEL<BX,BY,2><<<grid,block,0,stream>>>(src,src_cols,src_rows,src_pitch,dst,dst_cols,dst_rows,params,config,num_images); \
 else             KERNEL<BX,BY,3><<<grid,block,0,stream>>>(src,src_cols,src_rows,src_pitch,dst,dst_cols,dst_rows,params,config,num_images); \
}while(0)

// Enumerate supported (block.x, block.y) pairs once
#define LB_EACH_BXY(CALL_MACRO,KERNEL) \
 else if(block.x==32 && block.y==8){ CALL_MACRO(KERNEL,32,8); } \
 else if(block.x==16 && block.y==16){ CALL_MACRO(KERNEL,16,16); } \
 else if(block.x==64 && block.y==4){ CALL_MACRO(KERNEL,64,4); } \
 else if(block.x==32 && block.y==16){ CALL_MACRO(KERNEL,32,16); } \
 else if(block.x==64 && block.y==8){ CALL_MACRO(KERNEL,64,8); } \
 else if(block.x==48 && block.y==8){ CALL_MACRO(KERNEL,48,8); } \
 else if(block.x==24 && block.y==16){ CALL_MACRO(KERNEL,24,16); } \
 else if(block.x==40 && block.y==8){ CALL_MACRO(KERNEL,40,8); } \
 else if(block.x==20 && block.y==16){ CALL_MACRO(KERNEL,20,16); }

#define LB_DISPATCH_TABLE(CALL_MACRO,KERNEL) do{ \
 if(0){} LB_EACH_BXY(CALL_MACRO,KERNEL) else{/* generic by caller */} \
}while(0)

#define LB_DISPATCH_TABLE_OR_GENERIC(CALL_MACRO,KERNEL,...) do{ \
 if(0){} LB_EACH_BXY(CALL_MACRO,KERNEL) else{ __VA_ARGS__; } \
}while(0)
