#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import os
import pandas as pd


spam_file = '00_SMSSpamCollection'

if not os.path.isfile(spam_file):
    print(spam_file, ' is missing.')
    exit()

spam_list = []

spam = 'spam'
ham = 'ham'

with open(spam_file, encoding='utf-8') as fp:
    spam_lines = fp.readlines()
    count = 0
    for spam_line in spam_lines:
        if spam_line[:4] == spam:
            count += 1
            spam_list.append({'label': spam, 'message': spam_line[4:].strip()})
        elif spam_line[:3] == ham:
            count += 1
            spam_list.append({'label': ham, 'message': spam_line[3:].strip()})
        else:
            print(spam_line)

    total_count = len(spam_lines)
    print('Total spam lines - ', total_count)

    if total_count == count:
        print('All lines are validated.')

sms_df = pd.DataFrame(spam_list, columns=['label', 'message'])

print(sms_df.shape)
print(sms_df.describe())

sms_df.to_csv('sms.csv', encoding='utf-8', index=False, sep='\t')
