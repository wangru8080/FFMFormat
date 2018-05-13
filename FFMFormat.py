import pandas

def FFMFormat(df, label, path, category_feature = [], continus_feature = [], vector_feature = []):
    index = df.shape[0]
    data = open(path, 'w')
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

        for j, feat in enumerate(continus_feature):
            t = feat + '_' + str(df[feat][i])
            if t not in  feat_index.keys():
                feat_index[t] = feature_index
                feature_index = feature_index + 1
            feats.append('%s:%s:%s' % (field_index, feat_index[t], df[feat][i]))
            field_index = field_index + 1

        for j, feat in enumerate(vector_feature):
            words = df[feat][i].split(' ')
            for word in words:
                t = feat + '_' + word
                if t not in feat_index.keys():
                    feat_index[t] = feature_index
                    feature_index = feature_index + 1
                feats.append('%s:%s:%s' % (field_index, feat_index[t], 1))
            field_index = field_index + 1
        print('%s %s' % (df[label][i], ' '.join(feats)))
        data.write('%s %s\n' % (df[label][i], ' '.join(feats)))
    data.close()
