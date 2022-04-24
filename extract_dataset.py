import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Dataset info:
# PhysioNet: Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
# Dataset: Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.

# 1. Baseline, eyes open
# 2. Baseline, eyes closed
# 3. Task 1 (open and close left or right fist)
# 4. Task 2 (imagine opening and closing left or right fist)
# 5. Task 3 (open and close both fists or both feet)
# 6. Task 4 (imagine opening and closing both fists or both feet)
# 7. Task 1
# 8. Task 2
# 9. Task 3
# 10. Task 4
# 11. Task 1
# 12. Task 2
# 13. Task 3
# 14. Task 4

# Defines which tasks to use
TASK_ONE_REAL = [3, 7, 11]
TASK_TWO_IMAGINE = [4, 8, 12]

NUM_SUBJECTS = 109
TIME_END = 28959

# List of the data paths and labels to be populated (initial values will be column names)
LEGEND_REAL = {"data": "label"}
LEGEND_IMAGINE = {"data": "label"}


def prepareSubjectData(NUM_SUBJECTS):

    for i in range(1, NUM_SUBJECTS + 1):
        num = '{:03d}'.format(i)  # Formats the number to have at least 3 digits (eg 001, 002, etc.)
        subjectId = ("S{number}".format(number=num))
        getSubjectTask(subjectId, 1)  # Task one - subjects really open and close fists
        getSubjectTask(subjectId, 2)  # Task two - subjects imagine opening and closing fists

        # Progress bar for preparing dataset
        progress = i/NUM_SUBJECTS
        print("Progress: {progress_num}%".format(progress_num=round(progress * 100, 2)))


def getSubjectTask(subId, taskNum):
    if taskNum == 1:
        task_list = TASK_ONE_REAL
        task_type = "REAL"
    else:
        task_list = TASK_TWO_IMAGINE
        task_type = "IMAGINE"

    for i in task_list:
        formatted_task_num = '{:02d}'.format(i)
        pathName = "./files/{subId}/{subId}R{task_num}.edf".format(subId = subId, task_num = formatted_task_num)
        data = mne.io.read_raw_edf(pathName)
        info = data.info
        # Events define what sort of data is being recorded eg. resting, right hand, left hand
        events = mne.events_from_annotations(data)
        sampling_freq = info['sfreq']

        separateEvents(subId, events, data, sampling_freq, task_type)


def separateEvents(subId, events, raw_data, sampling_freq, task_type):
    # Separates the events into their respective states (rest, left hand, right hand) and start/end time

    events = events[0]
    for i in range(len(events)):
        # Finds the code, representing what action the subject is performing
        code = events[i][2]  # 1 == rest, 2 == left hand open and close, 3 == right hand open and close

        # Separates the events to be sorted by code, finds start and end time of event to slice data
        if i < len(events) - 1:
            event_time_start = events[i][0]
            event_time_end = events[i+1][0] - 1
        else:
            event_time_start = events[i][0]
            event_time_end = TIME_END

        data = sliceEventData(raw_data, event_time_start, event_time_end)
        exportSliceToCSV(raw_data, data, code, subId, i, task_type)


def sliceEventData(data, time_start, time_end):
    # Slices the raw data into the different labels

    mv, time = data[:64, time_start:time_end]  # Select all 64 channels, between the start and end time of the event
    return mv


def exportSliceToCSV(raw, data, label, subId, i, task_type):
    # Exports a csv based on the type and label of a task

    csvName = "./data_extracted/{type}/{subjectId}_{iteration}.csv".format(type=task_type,
                                                                       subjectId=subId,
                                                                       iteration=i)
    if task_type == "REAL":
        LEGEND_REAL[csvName] = label
    elif task_type == "IMAGINE":
        LEGEND_IMAGINE[csvName] = label

    header = ", ".join(raw.ch_names)
    feature_array = []

    for i in range(len(raw.ch_names)):
        feature_data = data[i]
        feature_array.append(feature_data)

    df = pd.DataFrame(data)

    df.T.to_csv(csvName, index=False, header=raw.ch_names, sep=",")
    

def exportLegendToCSV(legend):
    # Exports legend data to csv for defining labels

    print("Creating legend CSV...")
    if legend == LEGEND_REAL:
        csvName = "./data_extracted/LEGEND_REAL.csv"
    else:
        csvName = "./data_extracted/LEGEND_IMAGINE.csv"

    # Converting dictionary of legend to numpy array for CSV export
    result = legend.items()
    data = list(result)
    convertedArray = np.array(data)
    np.savetxt(csvName, convertedArray, fmt='%s', delimiter=", ")


def main():
    prepareSubjectData(NUM_SUBJECTS)
    exportLegendToCSV(LEGEND_REAL)
    exportLegendToCSV(LEGEND_IMAGINE)


if __name__ == "__main__":
    main()
