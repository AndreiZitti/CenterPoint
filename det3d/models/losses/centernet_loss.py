import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import netcomp as nc
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat
import open3d as o3d
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target,cars):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float().unsqueeze(2) 
    pred1 = (pred*mask).detach().cpu().clone().numpy()
    target1 =  (target*mask).detach().cpu().clone().numpy()
    if cars:
      predsDimensionsx = pred1[:,:,0]
      predsDimensionsy = pred1[:,:,1]
      gtDimensionsx = target1[:,:,0]
      gtDimensionsy = target1[:,:,1]
      gtTuples = ()
      predsTuples = ()
      for i in range(len(gtDimensionsx[0])):
        if (predsDimensionsx[0][i] + predsDimensionsy[0][i] != 0 or gtDimensionsx[0][i] + gtDimensionsy[0][i]!=0 ):
          gtTuples = gtTuples + ([gtDimensionsx[0][i], gtDimensionsy[0][i], 0],)
          predsTuples = predsTuples + ([predsDimensionsx[0][i], predsDimensionsy[0][i],0],)
      predsTuples = np.asarray(predsTuples)
      gtTuples = np.asarray(gtTuples)
      pcd1 = o3d.geometry.PointCloud()
      pcd2 = o3d.geometry.PointCloud()
      pcd1.points = o3d.utility.Vector3dVector(predsTuples)
      pcd2.points = o3d.utility.Vector3dVector(gtTuples)
      lines1 = self.get_graph_knn(pcd1)
      lines2 = self.get_graph_knn(pcd2)
      positions = self.assign_closest(pcd2.points, pcd1.points)
      ourLoss1 = self.laplacianLoss(np.asarray(lines1.lines),np.asarray(lines2.lines))
      ourLoss2 = abs(self.triangleLoss(np.asarray(lines1.lines) ) - self.triangleLoss(np.asarray(lines2.lines)))
      connectivityGT, connectivityPreds = self.getConnectivity(lines2.lines, lines1.lines, len(lines2.points), len(lines1.points),
                                                            positions)
                                                            
      ourLoss3 = self.connectivityLoss(connectivityGT, connectivityPreds, positions, 7, 1)
      ourLoss3 = np.mean(ourLoss3)
      
      ourLoss = ourLoss1 + ourLoss2 + ourLoss3
    else:
      ourLoss = 0

    loss = F.l1_loss(pred*mask, target*mask, reduction='none')
    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss, ourLoss

  def getConnectivity(self,gtTuples, predsTuples,gtLen, predsLen,positions):
    connectivityGT1 = [0] * gtLen
    for i in gtTuples:
        connectivityGT1[i[0]] += 1
        connectivityGT1[i[1]] += 1

    connectivityPreds1 = [0] * predsLen
    for i in predsTuples:
        connectivityPreds1[i[0]] += 1
        connectivityPreds1[i[1]] += 1

    for i in range(len(connectivityPreds1)):
        if(connectivityPreds1[i]==0):
            connectivityPreds1[i] = -50


    for i in range(len(connectivityGT1)):
        if(connectivityGT1[i]==0):
            connectivityGT1[i] = -50

    return connectivityGT1, connectivityPreds1

  def get_graph_knn(self,pcd):
    ind_global = []
    for i in pcd.points:
        i[2] = 0
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for idx in range(len(pcd.points)):
      [k, indices, _] = pcd_tree.search_knn_vector_3d(pcd.points[idx], 3)
      ind_local = []
      for jdx in range(len(indices)):
          ind_local.append(indices[jdx])
      ind_global.append(ind_local)
    graph = []
    for idx in range(1):
      local_graph = self.draw_local_graph(pcd, ind_global)
      graph.append(local_graph)
    return local_graph

  def draw_local_graph(self,poi, indices,color=[1, 0.5, 0]):
    lines = []
    for idx in range(len(indices)):
      lines.append([indices[idx][0], indices[idx][1]])
      if(len(indices[idx])>2):
        lines.append([indices[idx][0], indices[idx][2]])

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=poi.points,
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set
  
  def laplacianLoss(self,edges1,edges2):
    graph1 = nx.Graph()
    graph2 = nx.Graph()
    for tuple in edges1:
        graph1.add_edge(tuple[0],tuple[1])
    for tuple in edges2:
        graph2.add_edge(tuple[0],tuple[1])
    a1, a2 = [nx.adjacency_matrix(graph) for graph in [graph1, graph2]]
    return nc.lambda_dist(a1,a2,kind='laplacian',k=10)
  
  def triangleLoss(self,edges):
    graph = nx.Graph()
    for tuple in edges:
        graph.add_edge(tuple[0],tuple[1])

    return sum(nx.triangles(graph).values()) / 3
  
  def connectivityLoss(self,gt,preds,positions,FP,FN):
    indexesPreds = [0] * len(preds)
    connectivityPreds = [-1] * len(preds)
    for i in range(len(positions)):
        if(positions[i]>=0):
            connectivityPreds[i] = abs(gt[i] - preds[positions[i]])
            indexesPreds[positions[i]]+=1
        else:
            connectivityPreds[i] = FN #redundant

    for i in range(len(preds)):
        if(indexesPreds[i]==0):
            connectivityPreds.append(FP)

    return connectivityPreds
  def assign_closest(self,gtPoints,predsPoints):
    dist_mat =np.asarray(distance_matrix(gtPoints,predsPoints,p=2))
    positions = []
    r = 0.5
    check = True
    for i in range(len(gtPoints)):
        minV = min(dist_mat[i])
        pos = np.where(dist_mat[i] == minV)
        pos =  pos[0].item(0)
        positions.append(pos)
        check = True
        exitLopp = 0
        while check:
            if(len(positions)==len(set(positions))):
                check = False
            else:
                exitLopp +=1
                p1,p2 = self.findPositions(positions)
                positions = self.solveConflict(dist_mat, p1, p2, positions,r)
            if exitLopp==11:
              check=False
              # print("EXITLOOP")
              # print("EXITLOOP")
              # print("EXITLOOP")
    return positions
  def findPositions(self,positions):
    for i in range(len(positions)):
        val = positions[i]
        for j in range(i+1,len(positions)):
            if(positions[j]==val):
                return i,j

  def solveConflict(self,dist_mat, p1,p2,positions,r):
    minDist1 = dist_mat[p1]
    minDist2 = dist_mat[p2]
    check = True
   
    #the minimum distance to nearest gt
    min1 = min(minDist1)
    min2 = min(minDist2)
    #the id of the groundtruth

    if(min1>min2):
        pos1 = np.where(minDist1 == min1)
        pos = pos1[0].item(0)
        minDist1[pos] = 100
        min1 = min(minDist1)
        if(min1>r):
            positions[p1] = -100 - min1
        else:
            pos1 = np.where(minDist1 == min1)
            pos = pos1[0].item(0)
            positions[p1] = pos
    if (min2 > min1):
        pos2 = np.where(minDist2 == min2)
        pos = pos2[0].item(0)
        minDist2[pos] = 100
        min2 = min(minDist2)
        if(min2>r):
            positions[p2] = -100 - min2
        else:
            pos2 = np.where(minDist2 == min2)
            pos = pos2[0].item(0)
            positions[p2] = pos

    return positions
class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    mask = mask.float()
    gt = torch.pow(1 - target, 4)
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    neg_loss = neg_loss.sum()

    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos
