import pandas
import hashlib 

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16) % (nr_bins-  1) + 1

def gen_hashed_fmm_feats(feats, nr_bins = int(1e+6)):
    feats = ['%s:%s:%s' %(field, hashstr(feat, nr_bins), value) for (field, feat, value) in feats]
    return feats

def FFMFormat_hash(df, label, path, category_feature = [], continuous_feature = [], vector_feature = []):
    index = df.shape[0]
    data = open(path, 'w')
    feature_index = 0
    for i in range(index):
        feats = []
        field_index = 0
        for j, feat in enumerate(category_feature):
            feats.append((field_index, feat + '_' + str(df[feat][i]), 1))
            field_index = field_index + 1
            # feature_index = feature_index + 1
        for j, feat in enumerate(continus_feature):
            feats.append((field_index, feat + '_' + str(df[feat][i]), df[feat][i]))
            field_index = field_index + 1
            # feature_index = feature_index + 1
        for j, feat in enumerate(vector_feature):
            words = df[feat][i].split(' ')
            for word in words:
                feats.append((field_index, feat + '_' + word, 1))
            field_index = field_index + 1
        feats = gen_hashed_fmm_feats(feats)
        print('%s %s' % (df[label][i], ' '.join(feats)))
        data.write('%s %s\n' % (df[label][i], ' '.join(feats)))
    data.close()
    
FFMFormat_hash(df, 'label', '../data/ffm/data.ffm', category_feature, continuous_feature, vector_feature)
