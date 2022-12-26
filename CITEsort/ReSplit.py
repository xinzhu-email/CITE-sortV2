#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:44:58 2020

@author: lianqiuyu
"""

import sys
sys.path.append("./CITEsort")

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal, norm
import itertools
from scipy import stats
import operator
from scipy.spatial import distance
from BTree import BTree
import copy
#from scipy.signal import upfirdn
#import pandas as pd
import random



def all_BIC(leaf_dict, n_features):
        ll, n_features, n_sample = 0, 0, 0
        for key,node in leaf_dict.items():
            ll = ll + node.ll * node.weight 
            # n_features = n_features + len(node.key)
            n_sample = n_sample + len(node.indices)
            # node_name.append(str(key))
        cov_params = len(leaf_dict) * n_features * (n_features + 1) / 2.0
        mean_params = n_features * len(leaf_dict)
        n_param = int(cov_params + mean_params + len(leaf_dict) - 1)
        bic_score = -2 * ll * n_sample + n_param * np.log(n_sample)

        return bic_score

def assign_GMM(sample, mean_list, cov_list, weight, if_log=False, marker_list=None):
    # print(cov_list)
    index = sample.index
    # sample = np.array(sample)
    if if_log:
        type_num = np.log(weight/sum(weight))
    else:
        type_num = weight/sum(weight)
    
    p_prior = np.zeros(shape=(len(sample),len(weight)))
    for i in range(len(weight)):
        if if_log:
            print(i)
            p_prior[:,i] = multivariate_normal.logpdf(np.array(sample.loc[:,marker_list]), mean=np.array(mean_list[i][marker_list]), cov=np.array(cov_list[i].loc[marker_list,marker_list]),allow_singular=True)
            p_prior[:,i] = p_prior[:,i] + type_num[i]
            
        else:
            
            # print([cov_list[i][j,j] for j in range(len(cov_list[i]))])
            # print(sample.loc[:,marker_list[i]])
            # print(mean_list[i])
            # print(cov_list[i])
            p_prior[:,i] = multivariate_normal.pdf(np.array(sample.loc[:,marker_list[i]]), mean=np.array(mean_list[i]), cov=np.array(cov_list[i]))   
            p_prior[:,i] = p_prior[:,i] * type_num[i]
    # p_prior = -p_prior 
    
    p_post = p_prior / (p_prior.sum(axis=1)[:,np.newaxis] )
    pred_label = np.argmin(p_post,axis=1)
    # print(p_prior[:10,:])
    # print(pred_label[:10])
    pred_label = pd.Series(data=pred_label,index=index)
    return pred_label

def Choose_leaf(leaf_dict=None,data=None,bic_list=[],leaf_list=None,n_features=0,merge_cutoff=0.1,max_k=10,max_ndim=2,bic='bic'):
    if leaf_dict == None:
        root=ReSplit(data,merge_cutoff,marker_set=[])
        leaf_dict = {0: root}
        root.ind = 0
        leaf_list = [root]
        
    ### _____Choose maxmum loglikely hood gain as new root_____
    max_ll, max_root, separable = 0, None, False
    # print(leaf_dict.items())
    for key in list(leaf_dict.keys()):
        node = leaf_dict[key]
        if node.stop != None:
            # leaf_dict.pop(node.ind)
            continue
        separable = True
        if node.score_ll >= max_ll:
            max_ll = node.score_ll
            max_root = node
        # else:
        #     leaf_dict.pop(node.ind)
        #     leaf_dict[node.ind] = node
    bic_min_node = leaf_list  
    if separable:
        print('node.key:',list(max_root.key))
        if list(max_root.key)[0] == 'leaf':
            marker_set = list(set(max_root.marker))
            
        else:
            marker_set = list(set(max_root.marker + list(max_root.key)))
        max_root.left = ReSplit(max_root.child_left, merge_cutoff, max_root.weight * max_root.w_l, max_k, max_ndim, bic, marker_set=marker_set)
        max_root.left.ind = max(leaf_dict.keys()) + 1
        leaf_dict[max_root.left.ind] = max_root.left

        max_root.right = ReSplit(max_root.child_right, merge_cutoff, max_root.weight * max_root.w_r, max_k, max_ndim, bic, marker_set=marker_set)
        max_root.right.ind = max(leaf_dict.keys()) + 1
        leaf_dict[max_root.right.ind] = max_root.right

        leaf_dict.pop(max_root.ind)
        leaf_list = [x for x in leaf_list if x!=max_root]
        leaf_list.append(max_root.left)
        leaf_list.append(max_root.right)

        # Update data division for all leaves in leaf_dict, where to update?
        mean_list = [node.mean for node in leaf_list] 
        cov_list = [node.cov for node in leaf_list]
        w_list = [node.weight for node in leaf_list] 
        print(w_list)
        # marker_list = [node.marker for node in leaf_dict.values()]
        # marker_list = [list(data.columns) for i in range(len(leaf_dict.values()))]
        marker_list = [node.marker[i] for node in leaf_list for i in range(len(node.marker))]
        # marker_list = []
        # for node in leaf_dict.values():
        #     marker_list.append(node.marker)
        marker_list = list(set(marker_list))

        print('marker_list:',marker_list)
        new_label = assign_GMM(data, mean_list, cov_list, w_list, marker_list=marker_list, if_log=True)
        print(new_label.value_counts())
        for i in range(len(leaf_list)):
            node = leaf_list[i]
            sub_data = data[new_label==i]
            # if node.key == ('leaf',):
            #     node.weight = len(sub_data)/len(data)
            #     node.stop = 'no separable features'
            #     continue
            if len(sub_data) < len(list(node.key)) + 1 or len(sub_data) < 5:
                node.weight = len(sub_data)/len(data) 
                node.stop = 'small size'                        
                continue
            
            ind = node.ind
            node = ReSplit(sub_data, merge_cutoff, len(sub_data)/len(data), max_k, max_ndim, bic, marker_set=node.marker, root=node)
            node.ind= ind

            leaf_dict.pop(node.ind)
            leaf_dict[node.ind] = node
            leaf_list[i] = node

            # if node.stop == None:                
            #     node = ReSplit(sub_data, merge_cutoff, len(sub_data)/len(data), max_k, max_ndim, bic, marker_set=node.marker)
            # else:
            #     node.weight = len(sub_data)/len(data)
            #     print('node.wight:', node.weight, ', cell num: ', len(sub_data))
            #     node.mean = sub_data.loc[:,node.marker].mean()
            #     node.cov = sub_data.loc[:,node.marker].cov()
                
            
            # node.weight = (len(node.child_left)+len(node.child_right))/len(data)
            # node.child_left = data[new_label==i]
            # node.child_right = data[new_label==len(leaf_dict)+i]
            # node.ll = GaussianMixture(1,covariance_type='full').fit(pd.concat([node.child_left,node.child_right],ignore_index=False)).lower_bound_ * node.weight
            
            # node.w_l = len(node.child_left)/(len(node.child_left)+len(node.child_right))
            # node.w_r = 1 - node.w_l

            # gmm_l = GaussianMixture(1,covariance_type='full').fit(node.child_left.loc[:,node.marker])
            # node.mean_l, node.cov_l = gmm_l.means_, gmm_l.covariances_[0]
            # ll_l = gmm_l.lower_bound_ * node.w_l

            # gmm_r = GaussianMixture(1,covariance_type='full').fit(node.child_right.loc[:,node.marker])
            # node.mean_r, node.cov_r = gmm_r.means_, gmm_r.covariances_[0]
            # ll_r = gmm_r.lower_bound_ * node.w_r
            
            # node.score_ll = (ll_l + ll_r)*node.weight - node.ll
            # # likelihood also update?
        
        n_features = n_features + len(max_root.key)
        bic_score = all_BIC(leaf_dict, n_features)
        bic_list.append(bic_score)
        
        node, bic_list, bic_min_node = Choose_leaf(leaf_dict=leaf_dict, data=data, bic_list=bic_list, leaf_list=leaf_list, n_features=n_features)
        if bic_score <= min(bic_list):
            bic_min_node = leaf_list
    return max_root, bic_list, bic_min_node


def ReSplit(data=None,merge_cutoff=0.1,weight=1,max_k=10,max_ndim=2,bic='bic',marker_set=None, root=None):

    if root == None:
        root = BTree(('leaf',))
    root.indices = data.index.values.tolist()
    root.weight = weight
    root.stop = None
    root.marker = marker_set
    
    # if root.marker == None:
    if True:
        root.mean = data.mean()
        root.cov = data.cov()
    else:
        root.mean = data.loc[:,root.marker].mean()
        root.cov = data.loc[:,root.marker].cov()
    #if len(root.indices) < 500:
    #    print(root.indices)

    if data.shape[0] < 2:        
        root.all_clustering_dic = _set_small_leaf(data)
        root.stop = 'small size'
        return root

    unimodal = GaussianMixture(1,covariance_type='full').fit(data)
    root.ll = root.weight * unimodal.lower_bound_
    root.bic = unimodal.bic(data)
    
    separable_features, bipartitions, scores_ll, bic_list, all_clustering_dic = HiScanFeatures(data,root,merge_cutoff,max_k,max_ndim,bic)

    if len(separable_features) == 0:
        root.all_clustering_dic = all_clustering_dic
        root.stop = 'no separable features'
        root.key = ('leaf',)
        return root

    '''
    scores_ll = np.zeros(len(separable_features))
    bic_list = np.zeros(len(separable_features))
    for fidx in range(len(separable_features)):
        f = separable_features[fidx]
        if np.sum(bipartitions[f]) < 2 or np.sum(~bipartitions[f]) < 2:
            continue
        gmm1 = GaussianMixture(1,covariance_type='full').fit(data.loc[bipartitions[f],:])
        ll1 = gmm1.lower_bound_ * sum(bipartitions[f])/len(bipartitions[f])
        bic1 = gmm1.bic(data.loc[bipartitions[f],:]) 
        
        gmm0 = GaussianMixture(1,covariance_type='full').fit(data.loc[~bipartitions[f],:])
        ll0 = gmm0.lower_bound_ * sum(~bipartitions[f])/len(bipartitions[f])
        bic0 = gmm0.bic(data.loc[~bipartitions[f],:]) 
        
        scores_ll[fidx] = (ll1 + ll0) * root.weight - root.ll
        bic_list[fidx] = bic1 + bic0
    '''
    #print(separable_features)
    #print(scores_ll)
    #print(bic_list)
    idx_best = np.argmax(scores_ll)
    if np.max(scores_ll) < 0.001:
    #if root.bic < bic_list[idx_best]:
        root.stop = 'spliting increases bic'
        return root

    #idx_best = np.argmax(scores_ent)
    best_feature = separable_features[idx_best]
    best_partition = bipartitions[best_feature]
    best_score = scores_ll[idx_best]
    #best_weights = all_clustering_dic[len(best_feature)][best_feature]['weight']

    ## construct current node  
    root.key = best_feature
    root.all_clustering_dic = all_clustering_dic
    root.score_ll = best_score
    # root.marker = list(set(root.marker + list(best_feature)))
    print('root.marker:',root.marker)
    #root.marker_summary = marker_summary
    #root.para = para

    ## branch cells, component with higher mean goes right.
    # p1_mean = data.loc[best_partition, best_feature].mean()
    # p2_mean = data.loc[~best_partition, best_feature].mean()
    
    ### Calculate mean, std and weight for all dimension
    p1_mean = data.loc[best_partition, :].mean()
    p2_mean = data.loc[~best_partition, :].mean()
    p1_cov = data.loc[best_partition, :].cov()
    p2_cov = data.loc[~best_partition, :].cov()
    

    flag = True
    if len(p1_mean) == 1:
        flag = p1_mean.values > p2_mean.values
    else:
        p1_cosine = sum(p1_mean)/np.sqrt(sum(p1_mean**2))
        p2_cosine = sum(p2_mean)/np.sqrt(sum(p2_mean**2))
        flag = p1_cosine > p2_cosine

    if flag:
        root.child_right = data.iloc[best_partition, :]
        root.w_r = sum(best_partition)/len(best_partition)
        root.mean_r = p1_mean
        root.cov_r = p1_cov
        root.child_left = data.iloc[~best_partition, :] 
        root.w_l = sum(~best_partition)/len(best_partition)
        root.mean_l = p2_mean
        root.cov_l = p2_cov
        root.where_dominant = 'right'
    else:
        root.child_right = data.iloc[~best_partition, :]
        root.w_r = sum(~best_partition)/len(best_partition)
        root.mean_r = p2_mean
        root.cov_r = p2_cov
        root.child_left = data.iloc[best_partition, :]
        root.w_l = sum(best_partition)/len(best_partition)
        root.mean_l = p1_mean
        root.cov_l = p1_cov
        root.where_dominant = 'left'


    ## recursion
    # root.left = ReSplit(child_left,merge_cutoff,weight * w_l,max_k,max_ndim,bic)
    # root.right = ReSplit(child_right,merge_cutoff,weight * w_r,max_k,max_ndim,bic)

    return root





def HiScanFeatures(data,root,merge_cutoff,max_k,max_ndim,bic):
    
    # Try to separate on one dimension
    ndim = 1
    all_clustering_dic = {}
    separable_features, bipartitions, scores, bic_list, all_clustering_dic[ndim] = ScoreFeatures(data,root,merge_cutoff,max_k,ndim,bic)
    
    if len(separable_features) == 0:

        rescan_features = []
        for item in all_clustering_dic[ndim]:
            val = all_clustering_dic[ndim][item]['similarity_stopped']
            if val > 0.1 and val < 0.5:
                rescan_features.append(item[0])
        
        for ndim in range(2,max_ndim+1):
            if len(rescan_features) < ndim:
                separable_features, bipartitions, scores, bic_list, all_clustering_dic[ndim] = ScoreFeatures(data,root,0.5,max_k,len(rescan_features),bic)
                break
            
            separable_features, bipartitions, scores,bic_list, all_clustering_dic[ndim] = ScoreFeatures(data,root,0.5,max_k,ndim,bic,rescan_features)
            if len(separable_features) >= 1:
                break
        
    return separable_features, bipartitions, scores, bic_list, all_clustering_dic
    


def ScoreFeatures(data,root,merge_cutoff,max_k,ndim,bic,rescan_features=None):
    
    if rescan_features != None:
        F_set = rescan_features
    else:
        F_set = data.columns.values.tolist() # Feature list
    
    all_clustering = {}
    separable_features = []
    bipartitions = {}
    scores = []
    bic_list = []
        
    for item in itertools.combinations(F_set, ndim): # When ndim=1, search one feature each time. When ndim=2, search two features each time.
        x = data.loc[:,item]
        all_clustering[item] = Clustering(x,merge_cutoff,max_k,bic)
    
    for item in all_clustering:
        if all_clustering[item]['mp_ncluster'] > 1:
            
            merged_label = all_clustering[item]['mp_clustering']
            labels, counts = np.unique(merged_label, return_counts=True)
            if len(counts) == 1 or np.min(counts) < 5:
                continue
            
            ll_gain = []#np.zeros(len(labels))
            bic_mlabels = []
            for mlabel in labels:
                assignment = merged_label == mlabel

                marker_list = root.marker + list(item) # Choose only marker to calculate loglikelyhood
                # print('marker_list:',marker_list)

                # print(data.columns)
                gmm1 = GaussianMixture(1,covariance_type='full').fit(data.loc[assignment,marker_list]) # All features used
                ll1 = gmm1.lower_bound_ * sum(assignment)/len(assignment)
                bic1 = gmm1.bic(data.loc[assignment,marker_list]) 
                
                gmm0 = GaussianMixture(1,covariance_type='full').fit(data.loc[~assignment,marker_list])
                ll0 = gmm0.lower_bound_ * sum(~assignment)/len(assignment)
                bic0 = gmm0.bic(data.loc[~assignment,marker_list]) 
                
                ll_gain.append(  (ll1 + ll0) * root.weight - root.ll  )
                bic_mlabels.append( bic1 + bic0 )
            
            best_mlabel_idx = np.argmax(ll_gain)
            best_mlabel = labels[best_mlabel_idx]
            
            bipartitions[item] = merged_label == best_mlabel
            scores.append( ll_gain[best_mlabel_idx] )
            separable_features.append(item)
            bic_list.append( bic_mlabels[best_mlabel_idx] )
            
            # bipartitions[item] = all_clustering[item]['max_ent_p']
            # scores.append(all_clustering[item]['max_ent'])
            
    return separable_features, bipartitions, scores, bic_list, all_clustering



def Clustering(x,merge_cutoff,max_k,bic):
    
    val,cnt = np.unique(x.values.tolist(),return_counts=True)
    
    if len(val) < 50:
        clustering = _set_one_component(x) 
   
    else:
    
        k_bic,_ = BIC(x,max_k,bic)
    
        if k_bic == 1:    
            # if only one component, set values
            clustering = _set_one_component(x)      
        else:
            
            bp_gmm = GaussianMixture(k_bic).fit(x)
            clustering = merge_bhat(x,bp_gmm,merge_cutoff)
            '''
            if clustering['mp_ncluster'] > 1:
    
                merged_label = clustering['mp_clustering']
                labels, counts = np.unique(merged_label, return_counts=True)
                
                per = counts/np.sum(counts)
                ents = [stats.entropy([per_i, 1-per_i],base=2) for per_i in per]
                clustering['max_ent'] = np.max(ents)
                best_cc_idx = np.argmax(ents)
                best_cc_label = labels[best_cc_idx]
                clustering['max_ent_p'] = merged_label == best_cc_label
            '''
    return clustering



def bhattacharyya_dist(mu1, mu2, Sigma1, Sigma2):
    Sig = (Sigma1+Sigma2)/2
    ldet_s = np.linalg.det(Sig)
    ldet_s1 = np.linalg.det(Sigma1)
    ldet_s2 = np.linalg.det(Sigma2)
    d1 = distance.mahalanobis(mu1,mu2,np.linalg.inv(Sig))**2/8
    d2 = 0.5*np.log(ldet_s) - 0.25*np.log(ldet_s1) - 0.25*np.log(ldet_s2)
    return d1+d2



def merge_bhat(x,bp_gmm,cutoff):

    clustering = {}
    clustering['bp_ncluster'] = bp_gmm.n_components
    clustering['bp_clustering'] = bp_gmm.predict(x)
    clustering['bp_pro'] = bp_gmm.weights_
    clustering['bp_mean'] = bp_gmm.means_
    clustering['bp_Sigma'] = bp_gmm.covariances_
    
    #clustering['last_pair_similarity'] = _get_last_pair_similarity_2D(x,bp_gmm)
    gmm = copy.deepcopy(bp_gmm) 
    
    mu = gmm.means_
    Sigma = gmm.covariances_
    weights = list(gmm.weights_)
    posterior = gmm.predict_proba(x)
    
    current_ncluster = len(mu)
    mergedtonumbers = [int(item) for item in range(current_ncluster)]

    merge_flag = True
    clustering['bhat_dic_track'] = {}
    merge_time = 0

    while current_ncluster > 1 and merge_flag:

        bhat_dic = {}

        for c_pair in itertools.combinations(range(current_ncluster), 2):
            m1 = mu[c_pair[0],:]
            m2 = mu[c_pair[1],:]
            Sigma1 = Sigma[c_pair[0],:,:]
            Sigma2 = Sigma[c_pair[1],:,:]
            bhat_dic[c_pair] = np.exp(-bhattacharyya_dist(m1, m2, Sigma1, Sigma2))

        clustering['bhat_dic_track'][merge_time] = bhat_dic
        merge_time = merge_time + 1
        
        max_pair = max(bhat_dic.items(), key=operator.itemgetter(1))[0]
        max_val = bhat_dic[max_pair]

        if max_val > cutoff:
            merged_i,merged_j = max_pair
            # update mergedtonumbers
            for idx,val in enumerate(mergedtonumbers):
                if val == merged_j:
                    mergedtonumbers[idx] = merged_i
                if val > merged_j:
                    mergedtonumbers[idx] = val - 1
                    
            # update parameters
            weights[merged_i] = weights[merged_i] + weights[merged_j]
            
            posterior[:,merged_i] = posterior[:,merged_i] + posterior[:,merged_j]
            
            w = posterior[:,merged_i]/np.sum(posterior[:,merged_i])
            mu[merged_i,:] = np.dot(w,x)# update                                 
            
            x_centered = x.apply(lambda xx: xx-mu[merged_i,:],1)
            Sigma[merged_i,:,:] = np.cov(x_centered.T,aweights=w,bias=1)

            del weights[merged_j]
            #weights = np.delete(weights,merged_j,0)
            mu = np.delete(mu,merged_j,0)
            Sigma = np.delete(Sigma,merged_j,0)
            posterior = np.delete(posterior,merged_j,1)
            current_ncluster = current_ncluster - 1

        else:
            merge_flag = False
    
    
    clustering['similarity_stopped'] = np.min(list(bhat_dic.values()))
    clustering['mp_ncluster'] = mu.shape[0]
    clustering['mergedtonumbers'] = mergedtonumbers
    clustering['mp_clustering'] = list(np.apply_along_axis(np.argmax,1,posterior))
    
    return clustering



def _set_small_leaf(data):
    all_clustering_dic = {}
    all_clustering_dic[1] = {}
    
    F_set = data.columns.values.tolist()
    all_clustering = {}

    for item in itertools.combinations(F_set, 1):
        x = data.loc[:,item]
        all_clustering[item] = _set_one_component(x)
    
    all_clustering_dic[1] = all_clustering
    
    return all_clustering_dic



def _set_one_component(x):
    
    clustering = {}
    clustering['bp_ncluster'] = 1
    clustering['bp_clustering'] = [0]*len(x)
    clustering['bp_pro'] = [1]
    clustering['bp_mean'] = np.mean(x)
    clustering['bp_Sigma'] = np.var(x)
    clustering['bhat_dic_track'] = {}
    clustering['similarity_stopped'] = 1
    clustering['mp_ncluster'] = 1
    clustering['mp_clustering'] = [0]*len(x)
    clustering['mergedtonumbers'] = [0]

    return clustering



def BIC(X, max_k = 10,bic = 'bic'):
    """return best k chosen with BIC method"""
    
    bic_list = _get_BIC_k(X, min(max_k,len(np.unique(X))))
    
    if bic == 'bic':   
        return min(np.argmin(bic_list)+1,_FindElbow(bic_list)),bic_list
    elif bic == 'bic_min':   
        return np.argmin(bic_list)+1,bic_list
    elif bic == 'bic_elbow':
        return _FindElbow(bic_list),bic_list

    
    
def _get_BIC_k(X, max_k):
    """compute BIC scores with k belongs to [1,max_k]"""
    bic_list = []
    for i in range(1,max_k+1):
        gmm_i = GaussianMixture(i).fit(X)
        bic_list.append(gmm_i.bic(X))
    return bic_list



def _FindElbow(bic_list):
    """return elbow point, defined as the farthest point from the line through the first and last points"""
    if len(bic_list) == 1:
        return 1
    else:
        a = bic_list[0] - bic_list[-1]
        b = len(bic_list) - 1
        c = bic_list[-1]*1 - bic_list[0]*len(bic_list)
        dis = np.abs(a*range(1,len(bic_list)+1) + b*np.array(bic_list) + c)/np.sqrt(a**2+b**2)
        return np.argmax(dis)+1


