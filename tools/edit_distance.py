#!/usr/bin/python
#encoding=utf-8
import sys
def edit_distance(source, target):
    m = len(source) + 1
    n = len(target) + 1

    if m == 0:
        return n - 1
    if n == 0:
        return m - 1

    matrix = [[0 for i in range(0,m)]for j in range(0,n)]

    for i in range(0,m):
        matrix[0][i] = i
    for j in range(0,n):
        matrix[j][0] = j

    for j in range(1,n):
        for i in range(1,m):
            if source[i - 1] == target[j - 1]:
                matrix[j][i] = matrix[j - 1][i - 1]
            else:
                matrix[j][i] = min(matrix[j - 1][i - 1], matrix[j][i - 1], matrix[j - 1][i]) + 1

    return matrix[n - 1][m - 1]

def parse_file(src_file, is_ground_truth = True):
    file_dict = {}
    fp = open(src_file)
    lines = fp.readlines()
    for line in lines:
        line = line.translate(None, " ")
        line = line.translate(None, "\t")
        line = line.translate(None, "\r\n")
        line = line.translate(None, '啰')
        line = line.translate(None, '瞭')

        if is_ground_truth == True:
            word= line.split(".jpg")
        else:
            word = line.split(".jpg:")
        img_file = word[0]
        result = word[1] #.decode('utf-8')
        file_dict[img_file] = result
    fp.close()

    return file_dict

if __name__ == '__main__':

    grount_truth_file = "/data/heneng/images/scene/text_ground_truth.txt" # scene
    # grount_truth_file = "/data/heneng/images/news_caption/right_news_caption.txt"
    ocr_file = "./result.txt"
    ocr_file1 = u"D:\\Work\\样本\\标注样本\\字幕\\result.txt"
    ocr_file_baidu = u"D:\\Work\\百度\\文字识别OCR接入BCE\\result-baidu.txt"
    ocr_file_3c_add_border = u"E:\\server\\result\\result-add-border-3-channels.txt"
    ocr_file_3c = u"E:\\server\\result\\result-3-channels.txt"
    ocr_file_baidu1 = u"E:\\server\\result\\result-baidu.txt"
    grount_truth_dict = parse_file(grount_truth_file)
    #ocr_dict = parse_file(ocr_file_3c_add_border, False)    # correct_num = 754, error_num = 799
    #ocr_dict = parse_file(ocr_file_3c, False)    # correct_num = 508, error_num = 1568

    news_caption_ocr_file = '/data/heneng/work/baidu/result-news-caption.txt'
    ocr_dict = parse_file(ocr_file, False)    # correct_num = 508, error_num = 1568

    total_num = 0
    correct_num = 0
    for key, value in ocr_dict.iteritems():
        total_num += 1
        if grount_truth_dict[key] != ocr_dict[key]:
            #print grount_truth_dict[key]
            #print ocr_dict[key]
            print "Not equal %s\t%s" %(grount_truth_dict[key] , ocr_dict[key])
            continue
        else:
            print "equal", grount_truth_dict[key], ocr_dict[key]
            correct_num += 1

    print "total_num = %d, correct_num = %d, %8f" %(total_num, correct_num, correct_num * 1.0 / total_num)

    total_chars_num = 0
    error_num = 0
    for key, value in ocr_dict.iteritems():
        total_chars_num += len(grount_truth_dict[key])
        distance = edit_distance(grount_truth_dict[key], ocr_dict[key])
        error_num += distance
        #print distance #, grount_truth_dict[key], ocr_dict[key]

    print "total_chars_num = %d, error_num = %d" %(total_chars_num, error_num)
   # dist = edit_distance(source, target)
   # print dist
