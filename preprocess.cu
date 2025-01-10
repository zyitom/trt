// #include "preprocess.h"


// const int NUM_BOX_ELEMENT = 12;


// //映射到原来的坐标
// static __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy) {
//     *ox = matrix[0] * x + matrix[1] * y + matrix[2];
//     *oy = matrix[3] * x + matrix[4] * y + matrix[5];
// }

// static __global__ void decode_kernel(float *predict, int num_bboxes, int num_classes, float confidence_threshold,
//                                             float *invert_affine_matrix, float *parray, int max_objects,
//                                             GridAndStride *grid_and_stride, float dw, float dh, float scale, int total_anchors) {


//         int idx = blockIdx.x * blockDim.x + threadIdx.x;
//         //int total_coordinates = 0;
//         if (idx >= num_bboxes) {
//             return;
//         }

//         //if (idx >= total_anchors) return;  // 确保线程索引不超过锚点总数

//         // 计算每个线程对应的锚点索引
//         int anchor_idx = idx;

//         float *pitem = predict + (11 + num_classes) * anchor_idx;
//         float objectness = pitem[8];
//         float blue = pitem[9];
//         float red = pitem[10];
//         float color_confidence = max(red, blue);

//         //int color_label = (red > blue) ? 9 : 0;
//         int color_label =  0;
//         //if (objectness < confidence_threshold) return;

//         float *class_confidence = pitem + 11;
//         float confidence = *class_confidence;
//         int label = 0;

//         const int grid0 = grid_and_stride[anchor_idx].grid0;
//         const int grid1 = grid_and_stride[anchor_idx].grid1;
//         const int stride = grid_and_stride[anchor_idx].stride;

//         float coordinates[4][2];
//         bool all_positive = true;
//         for (int i = 0; i < 4; i++) {
//             float x = ((predict[anchor_idx * (11 + num_classes) + i * 2] + grid0) * stride - dw) / scale;
//             float y = ((predict[anchor_idx * (11 + num_classes) + i * 2 + 1] + grid1) * stride - dh) / scale;
//             coordinates[i][0] = x;
//             coordinates[i][1] = y;
//             if (x < 0 || y < 0) {
//                 all_positive = false;
//                 break;
//             }
//         }

//         if (!all_positive) return;

//         for (int i = 0; i < num_classes; ++i) {
//             float class_conf = class_confidence[i] * objectness;
//             if (class_conf > confidence) {
//                 confidence = class_conf;
//                 label = i;
//             }
//         }

//         if (objectness < 0.65) return;

//         float final_confidence = objectness * ((color_confidence + confidence) / 2);
//         //if (final_confidence < 0.6) return;

//         float ox1, oy1, ox2, oy2, ox3, oy3, ox4, oy4;

//         ox1 = coordinates[0][0];
//         oy1 = coordinates[0][1];
//         ox2 = coordinates[1][0];
//         oy2 = coordinates[1][1];
//         ox3 = coordinates[2][0];
//         oy3 = coordinates[2][1];
//         ox4 = coordinates[3][0];
//         oy4 = coordinates[3][1];

//         if (ox1 < 0 || oy1 < 0 || ox2 < 0 || oy2 < 0 || ox3 < 0 || oy3 < 0 || ox4 < 0 || oy4 < 0) return;

//         int index = atomicAdd(parray, 1);
//         if (index >= max_objects) return;

//         float *out_ptr = parray + 1 + index * 12;
//         out_ptr[0] = ox1;
//         out_ptr[1] = oy1;
//         out_ptr[2] = ox2;
//         out_ptr[3] = oy2;
//         out_ptr[4] = ox3;
//         out_ptr[5] = oy3;
//         out_ptr[6] = ox4;
//         out_ptr[7] = oy4;
//         out_ptr[8] = objectness;
//         out_ptr[9] = final_confidence;
//         out_ptr[10] =label + color_label;
//         out_ptr[11] = 1;

//     //    printf("Box #%d - Grid: (%d, %d) with Stride: %d, Coordinates: (%f, %f, %f, %f, %f, %f, %f, %f), Keep flag: %d\n",
//     //           index, grid0, grid1, stride,
//     //           out_ptr[0], out_ptr[1], out_ptr[2], out_ptr[3], out_ptr[4], out_ptr[5], out_ptr[6], out_ptr[7], (int)out_ptr[11]);
//     }
//     static __device__ float box_iou(
//         float ax1, float ay1, float ax2, float ay2, float ax3, float ay3, float ax4, float ay4,
//         float bx1, float by1, float bx2, float by2, float bx3, float by3, float bx4, float by4
// ){
//     float aleft = min(min(ax1, ax2), min(ax3, ax4));
//     float atop = min(min(ay1, ay2), min(ay3, ay4));
//     float aright = max(max(ax1, ax2), max(ax3, ax4));
//     float abottom = max(max(ay1, ay2), max(ay3, ay4));

//     float bleft = min(min(bx1, bx2), min(bx3, bx4));
//     float btop = min(min(by1, by2), min(by3, by4));
//     float bright = max(max(bx1, bx2), max(bx3, bx4));
//     float bbottom = max(max(by1, by2), max(by3, by4));
    
//     float cleft = max(aleft, bleft);
//     float ctop = max(atop, btop);
//     float cright = min(aright, bright);
//     float cbottom = min(abottom, bbottom);
//     float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
//     if(c_area == 0.0f) {

//         return 0.0f;
//     }

//     float a_area = (aright - aleft) * (abottom - atop);
//     float b_area = (bright - bleft) * (bbottom - btop);

//     float iou = c_area / (a_area + b_area - c_area);
//     return iou;
// }


// __global__ void fast_nms_kernel(float *bboxes, int max_objects, float threshold) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int total_boxes = min((int) *bboxes, max_objects); 
//     if (idx >= total_boxes) return;

//     float *current_box = bboxes + 1 + idx * NUM_BOX_ELEMENT; 

//     if (current_box[9] < threshold) // If the confidence of the current box is below the threshold, skip it
//         return;

//     for (int i = 0; i < idx; ++i) { // Only check against previously processed boxes to avoid redundancy
//         float *other_box = bboxes + 1 + i * NUM_BOX_ELEMENT;
//         if (other_box[11] == 0) continue; // Skip boxes already marked for deletion

//         float iou = box_iou(
//                 current_box[0], current_box[1], current_box[2], current_box[3],
//                 current_box[4], current_box[5], current_box[6], current_box[7],
//                 other_box[0], other_box[1], other_box[2], other_box[3],
//                 other_box[4], other_box[5], other_box[6], other_box[7]
//         );

//         if (iou > threshold) {
//             current_box[11] = 0; // Mark this box to be ignored
//             return; // No need to compare further if this box is suppressed
//         }
//     }
// }
