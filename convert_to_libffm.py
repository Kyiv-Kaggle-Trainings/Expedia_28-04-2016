import csv
import numpy as np
import time

all_features = ['date_time', 'user_id', 'site_name', 'posa_continent', 'user_location_country',
       'user_location_region', 'user_location_city',
       'orig_destination_distance', 'is_mobile', 'is_package',
       'channel', 'srch_ci', 'srch_co', 'srch_adults_cnt',
       'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id',
       'srch_destination_type_id', 'cnt', 'hotel_continent',
       'hotel_country', 'hotel_market', 'hotel_cluster', 'is_booking']

# this guy is our target label
all_features.remove('hotel_cluster')

# TODO: these need transformation
raw_date_features = ['date_time', 'srch_ci', 'srch_co']
all_features_excluding_date = [item for item in all_features if item not in raw_date_features]

# 
# Poor mans fransformer (rich people use sklearn interfaces as loading all of the stuff to memory just works)
#
class FfmTransform:
    def __init__(self, limit=999999999, verbose = False, features_to_encode = all_features_excluding_date):
        self.feature_dict = {}
        self.features_to_encode = features_to_encode
        # all the other ones are considered categorical
        self.continuos_features = ['orig_destination_distance']
        # used for debug
        self.limit = limit
        self.verbose = verbose
        self.last_feature_index = 0
        self.row_index = 0

    # Learns field mapping. Advantage of this guy is doing that in one-pass so it's faster
    #
    # It differs from the example on the slides at the moment in terms of indexing features for multiple columns simultaneously
    # Problem of that is that when you add a new field, indexes change
    #
    # TODO: Is libFFM sensitive to missing features? We can just pad indexes enough for each of the fields :/
    #
    # http://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf
    def streaming_fit(self, train_filename):
        print 'Starting streaming learning'
        self.start_time = time.time()
        with open(train_filename, 'rb') as f:
            csv_reader = csv.DictReader(f) 

            self.row_index = 0
            self.next_notification_time = 10
            self.next_check_time = 100000
            for row in csv_reader:
                self.row_index += 1

                for field_index, field_name in enumerate(self.features_to_encode):
                    field_value = row[field_name]
                    if not field_value:
                        continue

                    is_continuous = field_name in self.continuos_features
                    feature_index = self.get_feature_index(field_name, field_index, field_value, is_continuous)
                    feature_value = field_value if is_continuous else 1

                self.report_progress()

                if self.row_index >= self.limit:
                    print '(!) Stopping early as row limit(%d) reached' % self.limit
                    break

        time_spent = time.time() - self.start_time
        print 'Time spent learning: %d seconds' % time_spent
        print 'Distinct features: %d' % len(self.feature_dict)

    def transform(self, src_filename, dest_filename, label = None, overrides={}):
        self.start_time = time.time()
        print 'Transforming %s --> %s' % (src_filename, dest_filename)
        prior_features_count = len(self.feature_dict)
        with open(src_filename, 'rb') as src, open(dest_filename, 'w') as dest:
            csv_reader = csv.DictReader(src)
            self.row_index = 0
            self.next_check_time = 100000
            for row in csv_reader:
                self.row_index += 1
                written_field_index = 0
                if not label is None:
                    dest.write(row[label])
                    written_field_index += 1

                for field_index, field_name in enumerate(self.features_to_encode):
                    field_value = overrides[field_name] if field_name in overrides else row[field_name]
                    if not field_value:
                        continue

                    is_continuous = field_name in self.continuos_features
                    feature_index = self.get_feature_index(field_name, field_index, field_value, is_continuous)
                    feature_value = field_value if is_continuous else 1

                    if (written_field_index > 0): dest.write(' ')
                    dest.write('%s:%s:%s' % (field_index, feature_index, feature_value))
                    written_field_index += 1

                dest.write('\n')
                self.report_transformation_progress()

                if self.row_index >= self.limit:
                    print '(!) Stopping early as row limit(%d) reached' % self.limit
                    break

        time_spent = time.time() - self.start_time
        unknown_features_count = len(self.feature_dict) - prior_features_count
        print 'Time spent transforming: %d seconds' % time_spent
        print 'Unknown features spotted while transforming: %d' % unknown_features_count

    def report_progress(self):
        if self.verbose and self.row_index >= self.next_check_time:
            time_spent = time.time() - self.start_time
            self.next_check_time = self.next_check_time + 100000
            if time_spent > self.next_notification_time:
                self.next_notification_time = self.next_notification_time + 10
                feature_count = len(self.feature_dict)
                features_per_row = 1.0 * feature_count / self.row_index
                print '...time spent learning: %04d seconds / %08d rows / %07d features(%s/row)' % (time_spent, self.row_index, feature_count, features_per_row)

    def report_transformation_progress(self):
        if self.verbose and self.row_index >= self.next_check_time:
            time_spent = time.time() - self.start_time
            self.next_check_time = self.next_check_time + 100000
            if time_spent > self.next_notification_time:
                self.next_notification_time = self.next_notification_time + 10
                print '...time spent transforming: %04d seconds / %08d rows' % (time_spent, self.row_index)

    def get_feature_index(self, field_name, field_index, field_value, is_continuous):
        synthetic_feature_name = self.synthetic_name(field_name, field_index, field_value, is_continuous)
        if not synthetic_feature_name in self.feature_dict:
            self.last_feature_index += 1
            feature_index = self.last_feature_index
            self.feature_dict[synthetic_feature_name] = feature_index
        else:
            feature_index = self.feature_dict[synthetic_feature_name]
        return feature_index

    def synthetic_name(self, field_name, field_index, field_value, is_continuous):
        # TODO: using field_index can save a few hundred megs of memory in here, although less convenient for debugging
        return field_name if is_continuous else field_name + '_' + field_value    

transform = FfmTransform(verbose=True)

transform.streaming_fit(train_filename = '../train.csv')
transform.transform(src_filename = '../train.csv', dest_filename = '../train.csv.ffm', label = 'hotel_cluster')
transform.transform(src_filename = '../test.csv', dest_filename = '../test.csv.ffm', overrides={'is_booking': '1', 'cnt': '0'})
