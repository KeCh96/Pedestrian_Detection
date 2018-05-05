import os
import operator
import shutil
import pandas as pd
from matplotlib import pyplot as plt

MINOVERLAP = 0.5
plt.switch_backend('agg')



def voc_ap(rec, prec):
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    ind = []
    for ind_1 in range(1, len(mrec)):
        if mrec[ind_1] != mrec[ind_1-1]:
            ind.append(ind_1)
    ap = 0.0
    for i in ind:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, plot_color, true_p_bar):
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    if true_p_bar != "":
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions', left=fp_sorted)
        plt.legend(loc='lower right')
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1):
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            if i == (len(sorted_values)-1):
                adjust_axes(r, t, fig, axes)

    fig.canvas.set_window_title(window_title)
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    init_height = fig.get_figheight()
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    top_margin = 0.15
    bottom_margin = 0.05
    figure_height = height_in / (1 - top_margin - bottom_margin)
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize='large')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close()


def eval_mAP(pred_file_path, gt_file_path, output_dir, draw_plot = False):
    '''
    Ground truth file gt_files.csv columns:  <file_id> <class_name> <bbox> <used>
    Prediction file Pred_file.csv columns:  <file_id> <class_name> <bbox> <confidence>

    where:
        <class_name>: "all" or any class in coco or voc. str
        <file_id>: filename. str
        <bbox>: "left top right bottom". str
        <used>: whether has been matched while evaluating mAP, default set False. bool
        <confidence>: possibility score. float
    '''

    all_pred = pd.read_csv(pred_file_path)
    all_gt = pd.read_csv(gt_file_path)

    gt_counter_per_class = {}
    for gt_class, gt in all_gt.groupby('class_name'):
        gt_counter_per_class[gt_class] = len(gt)
    gt_classes = list(gt_counter_per_class.keys())
    pred_counter_per_class = {}
    for pred_class, pred in all_pred.groupby('class_name'):
        pred_counter_per_class[pred_class] = len(pred)
    pred_classes = list(pred_counter_per_class.keys())
    n_classes = len(gt_classes)
    sum_AP = 0.0
    ap_dictionary = {}
    count_true_positives = {}

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    results_file = open(output_dir + 'mAP.txt', 'w')

    for class_index, class_name in enumerate(gt_classes):
        predictions_class = all_pred[all_pred['class_name'] == class_name]
        ground_class = all_gt[all_gt['class_name'] == class_name]
        count_true_positives[class_name] = 0
        tp, fp = [0] * len(predictions_class), [0] * len(predictions_class)

        for idx, prediction in predictions_class.iterrows():
            file_id = prediction["file_id"]
            ground_truth_data = ground_class[ground_class['file_id'] == file_id]
            ovmax, gt_match = -1, -1
            bb = [ float(x) for x in prediction["bbox"].split() ]

            for _, obj in ground_truth_data.iterrows():
                bbgt = [ float(x) for x in obj["bbox"].split() ]
                bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        ovmax = ov
                        gt_match = obj

            if ovmax >= MINOVERLAP:
                if not bool(gt_match["used"]):
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positives[class_name] += 1
                else: fp[idx] = 1
            else: fp[idx] = 1

        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]

        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec, prec)
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP  "
        rounded_prec = [ '%.2f' % elem for elem in prec ]
        rounded_rec = [ '%.2f' % elem for elem in rec ]
        ap_dictionary[class_name] = ap
        results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

        if draw_plot:  # Draw plot
            plt.plot(rec, prec, '-o')
            plt.fill_between(mrec, 0, mprec, alpha=0.2, edgecolor='r')
            fig = plt.gcf()
            fig.canvas.set_window_title('AP ' + class_name)
            plt.title('class: ' + text)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            axes = plt.gca()
            axes.set_xlim([0.0,1.0])
            axes.set_ylim([0.0,1.05])
            fig.savefig(output_dir + class_name + ".png")

    mAP = sum_AP / n_classes
    text = "\n mAP = {0:.2f}%\n".format(mAP*100)
    results_file.write(text)
    print(text)

    for class_name in pred_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    results_file.write("\n# Number of ground_truth objects per class\n")
    for class_name in sorted(gt_counter_per_class):
        results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    results_file.write("\n# Number of predicted objects per class\n")
    for class_name in sorted(pred_classes):
        n_pred = pred_counter_per_class[class_name]
        text = class_name + ": " + str(n_pred)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_pred - count_true_positives[class_name]) + ")\n"
        results_file.write(text)

    if draw_plot:
        draw_plot_func(dictionary = gt_counter_per_class,
                       n_classes = n_classes,
                       window_title = "Ground-Truth Info",
                       plot_title = "Ground-Truth (" + str(len(all_gt)) + " files and " + str(n_classes) + " classes)",
                       x_label = "Number of objects per class",
                       output_path = output_dir + "Ground-Truth Info.png",
                       plot_color = 'forestgreen',
                       true_p_bar = "")

        count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(pred_counter_per_class.values()))
        draw_plot_func(dictionary = pred_counter_per_class,
                       n_classes = len(pred_classes),
                       window_title = "Predicted Objects Info",
                       plot_title = "Predicted Objects (" + str(len(all_pred)) + " files and " + str(count_non_zero_values_in_dictionary) + " detected classes)",
                       x_label = "Number of objects per class",
                       output_path = output_dir + "Predicted Objects Info.png",
                       plot_color = 'forestgreen',
                       true_p_bar = count_true_positives)

        draw_plot_func(dictionary = ap_dictionary,
                       n_classes = n_classes,
                       window_title = "evaluate",
                       plot_title = "evaluate = {0:.2f}%".format(mAP*100),
                       x_label = "Average Precision",
                       output_path = output_dir + "evaluate.png",
                       plot_color = 'royalblue',
                       true_p_bar = "")


