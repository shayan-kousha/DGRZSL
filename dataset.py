import numpy as np
import scipy.io as sio
import pickle
from sklearn import preprocessing
import torch


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label

class LoadDataset_GBU(object):
    def __init__(self, opt, main_dir, is_val=False):
        if opt.dataset == 'imageNet1K':
            self.read_matimagenet(opt, main_dir)
        else:
            self.read_matdataset(opt, main_dir, is_val)

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.train_cls_num = self.train_seen_classes.shape[0]
        self.val_cls_num = self.val_unseen_classes.shape[0]
        self.test_cls_num = self.test_unseen_classes.shape[0]
        self.test_seen_cls_num = self.test_seen_classes.shape[0]
        self.tr_cls_centroid = np.zeros([self.train_seen_classes.shape[0], self.feature_dim], np.float32)  # .astype(np.float32)
        for i in range(self.train_seen_classes.shape[0]):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)


    def read_matimagenet(self, opt, main_dir):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long()
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long()
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long()
        self.ntrain = self.train_feature.size()[0]
        self.seen_classes = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseen_classes = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seen_classes.size(0)
        self.ntest_class = self.unseen_classes.size(0)

    def read_matdataset(self, opt, main_dir, is_val=False):
        matcontent = sio.loadmat(main_dir + "data/GBU/data/" + opt.dataset + "/res101.mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(main_dir + "data/GBU/data/" + opt.dataset + "/att_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        if not is_val:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.val_unseen_feature = torch.from_numpy(np.array([])).float()
                self.val_unseen_label = torch.from_numpy(np.array([])).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.val_unseen_feature = torch.from_numpy(np.array([])).float()
                self.val_unseen_label = torch.from_numpy(np.array([])).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[train_loc])
                _val_unseen_feature = scaler.fit_transform(feature[val_unseen_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[train_loc]).long()
                self.val_unseen_feature = torch.from_numpy(_val_unseen_feature).float()
                self.val_unseen_feature.mul_(1 / mx)
                self.val_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[train_loc]).float()
                self.train_label = torch.from_numpy(label[train_loc]).long()
                self.val_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
                self.val_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        self.train_seen_classes = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.val_unseen_classes = torch.from_numpy(np.unique(self.val_unseen_label.numpy()))
        self.test_unseen_classes = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.test_seen_classes = torch.from_numpy(np.unique(self.test_seen_label.numpy()))

        self.train_label = map_label(self.train_label, self.train_seen_classes)
        self.val_unseen_label = map_label(self.val_unseen_label, self.val_unseen_classes)
        self.test_unseen_label = map_label(self.test_unseen_label, self.test_unseen_classes)
        self.test_seen_label = map_label(self.test_seen_label, self.test_seen_classes)
        self.train_att = self.attribute[self.train_seen_classes].numpy()
        self.val_att = self.attribute[self.val_unseen_classes].numpy()
        self.test_att = self.attribute[self.test_unseen_classes].numpy()
        self.test_seen_att = self.attribute[self.test_seen_classes].numpy()

class LoadDataset(object):
    def __init__(self, opt, main_dir, is_val=True):
        txt_feat_path = main_dir + 'data/CUB2011/CUB_Porter_7551D_TFIDF_new.mat'
        if opt.splitmode == 'easy':
            train_test_split_dir = main_dir + 'data/CUB2011/train_test_split_easy.mat'
            pfc_label_path_train = main_dir + 'data/CUB2011/labels_train.pkl'
            pfc_label_path_test = main_dir + 'data/CUB2011/labels_test.pkl'
            pfc_feat_path_train = main_dir + 'data/CUB2011/pfc_feat_train.mat'
            pfc_feat_path_test = main_dir + 'data/CUB2011/pfc_feat_test.mat'
            if is_val:
                train_cls_num = 150
                val_cls_num = 10
                test_cls_num = 40
            else:
                train_cls_num = 150
                val_cls_num = 0
                test_cls_num = 50
        else:
            train_test_split_dir = main_dir + 'data/CUB2011/train_test_split_hard.mat'
            pfc_label_path_train = main_dir + 'data/CUB2011/labels_train_hard.pkl'
            pfc_label_path_test = main_dir + 'data/CUB2011/labels_test_hard.pkl'
            pfc_feat_path_train = main_dir + 'data/CUB2011/pfc_feat_train_hard.mat'
            pfc_feat_path_test = main_dir + 'data/CUB2011/pfc_feat_test_hard.mat'
            if is_val:
                train_cls_num = 160
                val_cls_num = 10
                test_cls_num = 30
            else:
                train_cls_num = 160
                val_cls_num = 0
                test_cls_num = 40

        if is_val:
            data_features = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)
            with open(pfc_label_path_test, 'rb') as fout:
                data_labels = np.array(pickle.load(fout, encoding="latin1"))

            self.test_unseen_feature = data_features[data_labels < test_cls_num]
            self.val_unseen_feature = data_features[data_labels >= test_cls_num]
            self.train_feature = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
            self.test_unseen_label = data_labels[data_labels < test_cls_num]
            self.val_unseen_label = data_labels[data_labels >= test_cls_num] - test_cls_num
            with open(pfc_label_path_train, 'rb') as fout:
                self.train_label = pickle.load(fout, encoding="latin1")

            self.train_att, text_features = get_text_feature(txt_feat_path, train_test_split_dir)  # Z_tr, Z_te
            self.test_att, self.val_att = text_features[:test_cls_num], text_features[
                                                                                             test_cls_num:]
            self.text_dim = self.train_att.shape[1]
        else:
            self.train_feature = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
            self.test_unseen_feature = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)
            # calculate the corresponding centroid.
            with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
                self.train_label = pickle.load(fout1, encoding="latin1")
                self.test_unseen_label = pickle.load(fout2, encoding="latin1")

            self.train_att, self.test_att = get_text_feature(txt_feat_path, train_test_split_dir)  # Z_tr, Z_te
            self.text_dim = self.train_att.shape[1]

        self.train_cls_num = train_cls_num  # Y_train
        self.val_cls_num = val_cls_num
        self.test_cls_num = test_cls_num  # Y_test
        self.feature_dim = self.train_feature.shape[1]

        # Normalize feat_data to zero-centered
        mean = self.train_feature.mean()
        var = self.train_feature.var()
        self.train_feature = (self.train_feature - mean) / var  # X_tr
        self.test_unseen_feature = (self.test_unseen_feature - mean) / var  # X_te

        self.tr_cls_centroid = np.zeros([self.train_cls_num, self.train_feature.shape[1]]).astype(np.float32)
        for i in range(self.train_cls_num):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i], axis=0)

class LoadDataset_NAB(object):
    def __init__(self, opt, main_dir, is_val=True):
        txt_feat_path = main_dir + 'data/NABird/NAB_Porter_13217D_TFIDF_new.mat'
        if opt.splitmode == 'easy':
            train_test_split_dir = main_dir + 'data/NABird/train_test_split_NABird_easy.mat'
            pfc_label_path_train = main_dir + 'data/NABird/labels_train.pkl'
            pfc_label_path_test = main_dir + 'data/NABird/labels_test.pkl'
            pfc_feat_path_train = main_dir + 'data/NABird/pfc_feat_train_easy.mat'
            pfc_feat_path_test = main_dir + 'data/NABird/pfc_feat_test_easy.mat'
            if is_val:
                train_cls_num = 323
                val_cls_num = 21
                test_cls_num = 60
            else:
                train_cls_num = 323
                val_cls_num = 0
                test_cls_num = 81
        else:
            train_test_split_dir = main_dir + 'data/NABird/train_test_split_NABird_hard.mat'
            pfc_label_path_train = main_dir + 'data/NABird/labels_train_hard.pkl'
            pfc_label_path_test = main_dir + 'data/NABird/labels_test_hard.pkl'
            pfc_feat_path_train = main_dir + 'data/NABird/pfc_feat_train_hard.mat'
            pfc_feat_path_test = main_dir + 'data/NABird/pfc_feat_test_hard.mat'
            if is_val:
                train_cls_num = 323
                val_cls_num = 21
                test_cls_num = 60
            else:
                train_cls_num = 323
                val_cls_num = 0
                test_cls_num = 81

        if is_val:
            data_features = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)
            with open(pfc_label_path_test, 'rb') as fout:
                data_labels = pickle.load(fout, encoding="latin1")

            self.test_unseen_feature = data_features[data_labels < test_cls_num]
            self.val_unseen_feature = data_features[data_labels >= test_cls_num]
            self.train_feature = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
            self.test_unseen_label = data_labels[data_labels < test_cls_num]
            self.val_unseen_label = data_labels[data_labels >= test_cls_num] - test_cls_num
            with open(pfc_label_path_train, 'rb') as fout:
                self.train_label = pickle.load(fout, encoding="latin1")

            self.train_att, text_features = get_text_feature(txt_feat_path, train_test_split_dir)  # Z_tr, Z_te
            self.test_att, self.val_att = text_features[:test_cls_num], text_features[
                                                                                             test_cls_num:]
            self.text_dim = self.train_att.shape[1]
        else:
            self.train_feature = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
            self.test_unseen_feature = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)

            with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
                self.train_label = pickle.load(fout1, encoding="latin1")
                self.test_unseen_label = pickle.load(fout2, encoding="latin1")

            self.train_att, self.test_att = get_text_feature(txt_feat_path, train_test_split_dir)  # Z_tr, Z_te
            self.text_dim = self.train_att.shape[1]

        self.train_cls_num = train_cls_num  # Y_train
        self.val_cls_num = val_cls_num
        self.test_cls_num = test_cls_num  # Y_test
        self.feature_dim = self.train_feature.shape[1]

        # Normalize feat_data to zero-centered
        mean = self.train_feature.mean()
        var = self.train_feature.var()
        self.train_feature = (self.train_feature - mean) / var
        self.test_unseen_feature = (self.test_unseen_feature - mean) / var

        self.tr_cls_centroid = np.zeros([train_cls_num, self.train_feature.shape[1]]).astype(np.float32)
        for i in range(train_cls_num):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i], axis=0)

class FeatDataLayer(object):
    def __init__(self, label, feat_data, opt):
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        # self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels': minibatch_label}
        return blobs

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs


def get_text_feature(dir, train_test_split_dir):
    train_test_split = sio.loadmat(train_test_split_dir)
    # get training text feature
    train_cid = train_test_split['train_cid'].squeeze()
    text_feature = sio.loadmat(dir)['PredicateMatrix']
    train_text_feature = text_feature[train_cid - 1]  # 0-based index

    # get testing text feature
    test_cid = train_test_split['test_cid'].squeeze()
    text_feature = sio.loadmat(dir)['PredicateMatrix']
    test_text_feature = text_feature[test_cid - 1]  # 0-based index
    return train_text_feature.astype(np.float32), test_text_feature.astype(np.float32)
