import numpy as np
import pandas as pd
import os
import sys

class PlateUtils:
    def choice_box(self, vertex, out_boxes, out_scores, out_classes, image_size):
        predict_pos=[]
        x=[vertex["lu_x"],vertex["ld_x"],vertex["ru_x"],vertex["rd_x"]]
        y=[vertex["lu_y"],vertex["ld_y"],vertex["ru_y"],vertex["rd_y"]]

        for (out_box, out_score, out_class) in zip(out_boxes,out_scores,out_classes):
            pos_dict = self.created_boxes_vertex_dict(out_box, image_size)
            x_pos = self.xpos_in_box(pos_dict, x)
            y_pos = self.ypos_in_box(pos_dict, y)

            if x_pos and y_pos:
                pos_dict.update({"score":out_score})
                predict_pos.append(pos_dict)
        if len(predict_pos) >= 2:
            max_score=0
            for target_box in predict_pos:
                if target_box["score"]>max_score:
                    max_score = target_box["score"]
                    return_box = target_box
                else:
                    return_box = None
            return return_box
        elif len(predict_pos) == 0:
            return None
        return predict_pos[0]

    def created_boxes_vertex_dict(self, out_box, size):
        top, left, bottom, right = out_box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(size[0], np.floor(right + 0.5).astype('int32'))
        box = [top, left, bottom, right]

        pos_dict={}
        pos_name = ["top", "left", "bottom", "right"]
        for pos,name in zip(box, pos_name):
            pos_dict[name] = pos
        return pos_dict

    def xpos_in_box(self, pos_dict, arr):
        for x_pos in arr:
            if (pos_dict["left"] < x_pos < pos_dict["right"]) == False:
                return False
        return True

    def ypos_in_box(self, pos_dict, arr):
        for y_pos in arr:
            if (pos_dict["top"] < y_pos < pos_dict["bottom"]) == False:
                return False
        return True

    def move_and_update(self, vertex, corner_name, position):
        move_pos = vertex[corner_name] - position
        vertex.at[corner_name] = move_pos
        return move_pos, vertex

    def number_plate_crop(self, image, vertex, predict_pos, file_name):
        import cv2
        import numpy as np
        tmp_image = np.asarray(image)
        ocv_image = tmp_image[:, :, ::-1].copy()

        move_lux, vertex = self.move_and_update(vertex, "lu_x", predict_pos["left"])
        move_ldx, vertex = self.move_and_update(vertex, "ld_x", predict_pos["left"])
        move_rux, vertex = self.move_and_update(vertex, "ru_x", predict_pos["left"])
        move_rdx, vertex = self.move_and_update(vertex, "rd_x", predict_pos["left"])
        move_luy, vertex = self.move_and_update(vertex, "lu_y", predict_pos["top"])
        move_ldy, vertex = self.move_and_update(vertex, "ld_y", predict_pos["top"])
        move_ruy, vertex = self.move_and_update(vertex, "ru_y", predict_pos["top"])
        move_rdy, vertex = self.move_and_update(vertex, "rd_y", predict_pos["top"])

        pts1 = np.float32([[move_lux,move_luy],[move_rux,move_ruy],[move_ldx,move_ldy],[move_rdx,move_rdy]])
        pts2 = np.float32([[0,0],[640,0],[0,320],[640,320]])
        #透視変換
        M = cv2.getPerspectiveTransform(pts1,pts2)
        rst = cv2.warpPerspective(ocv_image,M,(640,320))
        cv2.imwrite('/Users/naka345/Desktop/deeplearning/number_plate/output/number/'+file_name,rst)

        return vertex
