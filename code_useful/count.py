# Id, ProductId, UserId, ProfileName
# HelpfulnessNumerator, HelpfulnessDenominator
# Score, Time, Summary, Text

import csv
from utils import split_sentence, write_file

sentence_threshold = 5

if __name__ == '__main__':
    print('Program begin!')

    print('Data reading...', end='')
    data, head_info = {}, []
    with open("../datasets/amazon-fine-food-reviews/Reviews.csv", "r", encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for data_line in reader:
            # head line
            if reader.line_num == 1:
                head_info = data_line
                for item in head_info:
                    data[item] = []
            # data line
            else:
                assert len(head_info) == len(data_line)
                for index, item in enumerate(data_line):
                    data[head_info[index]].append(item)
    print('finished!')

    # write_file('summary.txt', data['Summary'])
    # write_file('text.txt', data['Text'])

    print('\nLength-information:')
    print('  Id_num: {}'.format(len(set(data['Id']))))
    print('  ProductID_num: {}'.format(len(set(data['ProductId']))))
    print('  UserID_num: {}'.format(len(set(data['UserId']))))
    print('  ProfileName_num: {}'.format(len(set(data['ProfileName']))))

    print('\nScore-information')
    score_info = data['Score']
    score_count = [0] * 6
    for item in score_info:
        score_count[int(item)] = score_count[int(item)] + 1
    score_rate = [item * 100 / len(set(data['Id'])) for item in score_count]
    print(score_rate[1: 6])

    print('\nHelpful-information')
    denominator_count = 0
    denominator_list = [0] * 21
    numerator_count = 0
    numerator_list = [0] * 21
    helpful_rate = 0
    for index, denominator_str in enumerate(data['HelpfulnessDenominator']):
        numerator = int(data['HelpfulnessNumerator'][index])
        denominator = int(denominator_str)
        if numerator < 21:
            numerator_list[numerator] += 1
        if denominator < 21:
            denominator_list[denominator] += 1

        if numerator > 0:
            numerator_count += 1
        if denominator > 0:
            denominator_count += 1
            helpful_rate += int(numerator) / int(denominator)

    print('  denominator_distribution: ', end='')
    print(denominator_list[:21])
    print('  denominator > 0 rate: {}%'.format(denominator_count / len(set(data['Id'])) * 100))

    print('  numerator_distribution: ', end='')
    print(numerator_list[:21])
    print('  numerator > 0 rate: {}%'.format(numerator_count / len(set(data['Id'])) * 100))

    print('  helpful_rate: {}%'.format(helpful_rate / denominator_count * 100))

    print('\nTime & Helpful')
    timemore_count = 0
    timemore_sum = 0
    timezero_count = 0
    timezero_sum = 0
    for index, denominator_str in enumerate(data['HelpfulnessDenominator']):
        denominator = int(denominator_str)
        timestamp = int(data['HelpfulnessNumerator'][index])
        if denominator > 0:
            timemore_count += 1
            timemore_sum += timestamp
        else:
            timezero_count += 1
            timezero_sum += timestamp
    print(timemore_count, timezero_count)
    print(timemore_sum / timemore_count - timezero_sum / timezero_count)
