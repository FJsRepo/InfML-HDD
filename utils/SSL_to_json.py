import os
import json

def get_dirs(root_anno, root_img, anno_files_json, x_coordinate):
    for root, dirs, files in os.walk(root_anno):
        for filename in files:
            path_anno = os.path.join(root_anno, filename)
            filename = filename[:-5]
            y_coordinate = []
            path_img = os.path.join(root_img, filename + ".png")
            with open(path_anno, 'r') as SSL_anno:
                lines = SSL_anno.readlines()
                x1 = 0.0
                x2 = 383.0
                y1 = float(lines[9])
                y2 = float(lines[13])
                k = (y2-y1)/(x2-x1)
                b = y1
            for i in x_coordinate:
                temp_y = k*i + b
                temp_y = int(temp_y)
                y_coordinate.append(temp_y)
            data = {
                "x_coordinate":x_coordinate,
                "y_coordinate":y_coordinate,
                "raw_file":path_img
            }
            json_data = json.dumps(data)
            with open(anno_files_json, 'a') as json_file:
                json_file.write(json_data + '\n')


if __name__ == "__main__":

    root_anno = "../data/test_set/clips/1_test/annotations"

    root_img = "clips/1_test/images"

    anno_files_json = "test_data_1_test.json"

    x_coordinate = [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 383]

    get_dirs(root_anno, root_img, anno_files_json, x_coordinate)

    # with open(anno_files, 'r') as anno_obj:
    #     lines = anno_obj.readlines()
    #     print(lines[0])