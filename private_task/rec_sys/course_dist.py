# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:39:34 2020

@author: luoyan011
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


os.chdir('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\some_task\\rec_sys')
current_path = os.getcwd()
file_list = os.listdir(current_path)

raw_data = pd.read_csv("student_enrollment_converted.csv")
course_overview = pd.read_csv("course_overviews.csv")
# Generate Course Count
course_dist = raw_data['item'].value_counts().to_frame()
course_dist.reset_index(level = 0, inplace = True)
course_dist = course_dist.rename(columns = {"index":"course_id","item":"count"})

# Bar Chart
def plot_course_bar(data, course, count, threshold):
    plt_data = data[data[count] >= threshold]
    y_pos = np.arange(len(plt_data))
    y_pos = y_pos[::-1]
    plt.barh(y_pos, plt_data[count])
    plt.yticks(y_pos, plt_data[course])
    plt.yticks([])
    plt.show()
plot_course_bar(course_dist, "course_id", "count", 0)
plot_course_bar(course_dist, "course_id", "count", 25000)

# Get top 6 courses
drop_course = course_dist[course_dist['count']>60000]
drop_course = drop_course.merge(course_overview[['course_id', 'display_name']], on='course_id', how = 'left')
pd.set_option("display.max.columns", None)
print(drop_course)

# Drop popular course and save it as csv file
drop_course_id = list(drop_course.course_id)
new_data = raw_data[~raw_data.item.isin(drop_course_id)]

print("The original dataset has", len(raw_data), "rows and new dataset (after remove all popular courses) has", len(new_data), "rows.")
