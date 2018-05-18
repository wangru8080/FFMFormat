import pandas

'''
pandas to FFM
df = train + test
df.ffm -> train.ffm & test.ffm 
'''

def FFMFormat(df, label, path, train_len, category_feature = [], continuous_feature = [], vector_feature = []):
    index = df.shape[0]
    train = open(path + 'train.ffm', 'w')
    test = open(path + 'test.ffm', 'w')
    feature_index = 0
    feat_index = {}
    for i in range(index):
        feats = []
        field_index = 0
        for j, feat in enumerate(category_feature):
            t = feat + '_' + str(df[feat][i])
            if t not in  feat_index.keys():
                feat_index[t] = feature_index
                feature_index = feature_index + 1
            feats.append('%s:%s:%s' % (field_index, feat_index[t], 1))
            field_index = field_index + 1

        for j, feat in enumerate(continuous_feature):
            #t = feat + '_' + str(df[feat][i])
            #if t not in  feat_index.keys():
            #    feat_index[t] = feature_index
            #    feature_index = feature_index + 1
            #feats.append('%s:%s:%s' % (field_index, feat_index[t], df[feat][i]))
            feats.append('%s:%s:%s' % (field_index, feature_index, df[feat][i]))
            feature_index = feature_index + 1
            field_index = field_index + 1

        for j, feat in enumerate(vector_feature):
            words = df[feat][i].split(' ') # split要根据实际情况去分割， 例如以a,b,c方式存储则改为split(',')
            for word in words:
                t = feat + '_' + word
                if t not in feat_index.keys():
                    feat_index[t] = feature_index
                    feature_index = feature_index + 1
                feats.append('%s:%s:%s' % (field_index, feat_index[t], 1))
            field_index = field_index + 1
        print('%s %s' % (df[label][i], ' '.join(feats)))
        if i < train_len:
            train.write('%s %s\n' % (df[label][i], ' '.join(feats)))
        else:
            test.write('%s\n' % (' '.join(feats)))
    train.close()
    test.close()
    
FFMFormat(df, 'label', '../data/ffm/', train_len, category_feature, continuous_feature, vector_feature)
