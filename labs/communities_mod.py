#!/usr/bin/env python
# coding: utf-8

# # **Build probability matrices from word counts**

# In[ ]:


def clean_Mwd_matrix(Mwd,words,documents):

  # remove elements that are too central, e.g., #covid19
  not_wanted = np.array(Mwd.sum(axis=1)).flatten()>Mwd.shape[1]/4
  text = "removing: " + " ".join(words[not_wanted])
  words = words[~not_wanted]
  Mwd = Mwd[~not_wanted,:]

  # remove documents and words with fewer than 2 links
  while True:

    # keep memory
    dim_old = Mwd.size
    # remove documents with less than 2 words
    wanted = np.array(Mwd.sum(axis=0)).flatten()>1
    Mwd = Mwd[:,wanted]
    documents = documents[wanted]

    # remove words in less than 2 documents
    not_wanted = np.array(Mwd.sum(axis=1)).flatten()<=1
    text = text + " " + " ".join(words[not_wanted])
    words = words[~not_wanted]
    Mwd = Mwd[~not_wanted,:]
    # exit criterion
    if (dim_old == Mwd.size): break

  # exit
  print(text)
  return Mwd, words, documents

def logg(x):
    y = np.log(x)
    y[x==0] = 0
    return y

def probability_matrices(Mwd, equalik = True, tform = False):

    if equalik: # documents equally likely
        Pwd = Mwd/Mwd.sum(axis=0).flatten()/Mwd.shape[1]
    else: # documents proportional to their length
        Pwd = Mwd/Mwd.sum()
    # TF-IDF format
    if (tform):
        iw = -logg(np.sum(Mwd>0,axis=1).flatten()/Mwd.shape[1])
        Pwd = sps.diags(np.array(iw)[0])*Pwd # TF-IDF form
        Pwd = Pwd/Pwd.sum() # normalize, treat it as Pwd
    # words and document matrices
    pd = Pwd.sum(axis=0).flatten()
    Pww = (Pwd/pd)*(Pwd.T)
    pw = Pwd.sum(axis=1).flatten()
    Pdd = (Pwd.T/pw)*Pwd
    # joint words and document matrix - documents first
    Paa = sps.hstack((sps.csr_matrix((Pwd.shape[1],Pwd.shape[1])),Pwd.T))
    Paa = sps.vstack((Paa,sps.hstack((Pwd,sps.csr_matrix((Pwd.shape[0],Pwd.shape[0]))))))
    Paa = Paa/2.0

    return Pwd, Pww, Pdd, Paa


# # **Define performance measures**

# In[ ]:


def nmi_function(A): # A = Pwc
    aw = A.sum(axis=1).flatten() # word probability
    ac = A.sum(axis=0).flatten() # class probability
    Hc = np.multiply(ac,-logg(ac)).sum() # class entropy
    A2 = ((A/ac).T/aw).T
    A2.data = logg(A2.data)
    y = (A.multiply(A2)).sum()/Hc
    return y

def modularity_function(A):
    y = A.trace()-(A.sum(axis=0)*A.sum(axis=1)).item()
    return y

def ncut_function(A):
    y = ((A.sum(axis=0)-A.diagonal())/A.sum(axis=0)).mean()
    return y

def my_pagerank(M,q,c=.85,it=60):
    r = q.copy() # ranking matrix, initialized to q (copy)
    for k in range(it): # slow cycle
      r = c*M.dot(r) + (1-c)*q
    return r

def infomap_function(v):
    y = -(v.data*logg(v.data/v.sum())).sum()
    return y

def infomap_rank(Pdd):
    # transition matrix
    pd = Pdd.sum(axis=0).flatten()
    M = sps.csr_matrix(Pdd/pd)
    # pagerank vector - faster than r = my_pagerank(M,q)
    G = ig.Graph.Adjacency((M > 0).toarray().tolist())
    G.es['weight'] = np.array(M[M.nonzero()])[0]
    r = G.pagerank(weights='weight')
    r = (sps.csr_matrix(np.array(r))).T

    return r

def infomap(C,Pdd,r):
    pd = Pdd.sum(axis=0).flatten()
    M = Pdd/pd # transition matrix
    # extract vectors
    z = C.T*sps.diags(r.toarray().flatten())
    q = sps.csr_matrix((1,z.shape[0]))
    c = .85
    for i in range(z.shape[0]):
      tmp = (C[:,i].transpose()*M)*z[i].transpose()
      q[0,i] = (1-(1-c)*C[:,i].sum()/M.shape[0])*z[i].sum()-c*tmp[0,0]
    # extract statistics
    y = infomap_function(q)
    for i in range(z.shape[0]):
      y += infomap_function(sps.hstack([z[i],sps.csr_matrix([[q[0,i]]])]))
    # normalize
    y = (y/infomap_function(pd))-1

    return y

def clustering_statistics(C,Pwd,Pdd,r):

    Pwc = Pwd*C # joint word + class probability
    NMI = nmi_function(Pwc)
    Pcc = C.T*Pdd*C # joint class + class probability
    Q = modularity_function(Pcc)
    Ncut = ncut_function(Pcc)
    Infomap = infomap(C,Pdd,r)

    return [NMI, Q, Ncut, Infomap]


# # **PLMP**

# In[ ]:


def plmp(Mdw,v):
  B1 = (Mdw/Mdw.sum(axis=0).flatten()).T
  B2 = (Mdw.T/Mdw.sum(axis=1).flatten()).T
  M = B1*B2 # mixing matrix
  q = B1*v # teleport vector
  r = my_pagerank(M,q)
  return r


# # **BERTopic**

# In[ ]:


import copy

def topics_to_C(topics):

  # extract community assignments
  C = sps.csr_matrix((len(topics),max(topics)+2))
  for i in range(C.shape[1]):
    C[np.array(topics)==(i-1),i] = 1

  # remove zero assignments
  C = C[:,np.unique(scipy.sparse.find(C)[1])]

  return C

def plot_community_patterns(C,nrows,ncols,refs):

  fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                          figsize=(20*nrows, 20*ncols),
                          subplot_kw={'xticks': [], 'yticks': []})

  for i in range(len(C)):
    # identify an order for rankings based on refs
    if ((i%ncols)==0):
      tmp0 = C[refs[0]+i]
      tmp0 = np.array([tmp0[i].argmax()+1 for i in range(tmp0.shape[0])])
      tmp1 = C[refs[1]+i]
      tmp1 = np.array([tmp1[i].argmax()+1 for i in range(tmp1.shape[0])])
      pos = np.argsort(tmp0+(tmp1/tmp1.max())/2)

    # plot matrices
    tmp = sps.csr_matrix(C[i]).astype(np.float32)
    tmp = tmp[pos,] # reorder
    M = tmp*(tmp.T)
    ax = axs.flat[i]
    ax.imshow(M.toarray(), cmap='viridis')


print('bertopic 1.11')

def bertopic_overwrite(bert_model_in,docs,C):
  bert_model = copy.deepcopy(bert_model_in)

  # build the documents dataframe: 'Document' + "Topic"
  documents = pd.DataFrame(docs,columns=['Document'])
  tmp = np.array([C[i].argmax() for i in range(C.shape[0])])
  documents["Topic"] = tmp

  # update topic assignment
  bert_model.topics_ = tmp.tolist()

  # build cf-idf values
  documents_per_topic = documents.groupby(['Topic'],
                    as_index=False).agg({'Document': ' '.join})
  c_tf_idf_, words = bert_model._c_tf_idf(documents_per_topic)
  bert_model.c_tf_idf_ = c_tf_idf_

  # extract words representations
  topic_representations_ = bert_model._extract_words_per_topic(words, documents)
  bert_model.topic_representations_ = topic_representations_
  bert_model.topic_labels_ = {key: f"{key}_" + "_".join([word[0] for word in values[:4]])
                              for key, values in
                              topic_representations_.items()}

  # exit
  return bert_model


# # **soft Louvain algorithm**

# In[ ]:


def soft_assign(a,v):
  if (a>=0):
    u = np.array(v.data)
    n = np.where(u==u.max())[0][0]
    u = np.zeros(u.shape)
    u[n] = 1
    return np.array(u)/u.sum()
  else:
    u = -np.array(v.data)/a
    g = np.sort(u)[::-1]
    z = np.cumsum(g)-np.append(np.array(range(1,len(g)))*g[1:len(g)],-np.Inf)
    n = np.where(z>=1)[0][0]
    la = ((g[0:n+1]).sum()-1)/(n+1)
    u = u-la
    u[u<0] = 0
    return np.array(u)/u.sum()


# In[ ]:


from random import shuffle

def my_soft_louvain(A, C_start=None, seed=None):
    """
    Find the best partition of a graph using the Louvain Community Detection
    Algorithm.

    References
    [1] Blondel, V.D. et al. Fast unfolding of communities in large networks.
        J. Stat. Mech 10008, 1-12(2008).
        https://doi.org/10.1088/1742-5468/2008/10/P10008
    [2] Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden:
        guaranteeing well-connected communities. Sci Rep 9, 5233 (2019).
        https://doi.org/10.1038/s41598-019-41695-z
    [3] Nicolas Dugué, Anthony Perez. Directed Louvain : maximizing modularity
        in directed networks. [Research Report] Université d’Orléans. 2015.
        hal-01231784. https://hal.archives-ouvertes.fr/hal-01231784
    """

    # initialize random seed
    np.random.seed(seed)
    # normalize matrix - otherwise it doesn't work -  read by rows
    A = sps.csr_matrix(A)/A.sum()
    # initialize the community assignment matrix to "each node is a community"
    C = sps.csr_matrix(sps.identity(A.shape[0], dtype='float'))
    # starting assignment
    if (C_start==None):
      C_start = sps.csr_matrix(sps.identity(A.shape[0], dtype='float'))

    # main loop for the different layers of Louvain
    while True:
        print([C.shape[0], C.shape[1]])
        # improve modularity in this layer
        Clayer, improvement = _my_one_level(A,C_start)
        # exit if no improvement
        if (improvement==False): break
        # otherwise update variables according to the new clusters
        C = C*Clayer
        A = Clayer.T*(A*Clayer)
        # initialize the community assignment matrix to "each node is a community"
        C_start = sps.csr_matrix(sps.identity(A.shape[0], dtype='float'))

    # return community assignments and resulting adjacency matrix
    Q = modularity_function(A)
    return C, A, Q

print("softlouvain v1.10")

def _my_one_level(A,C):

    N = A.shape[0] # number of nodes
    rand_nodes = list(range(N)) # random nodes list
    shuffle(rand_nodes) # shuffle random nodes list

    d_in = np.transpose(A.sum(axis=1)) # input degrees - row vector
    d_out = A.sum(axis=0) # output degrees - row vector
    A = A + A.T # sum easily accessible by row

    # main loop - loop until you do not see any improvement
    improvement = False
    while True:

        # counter for the number of nodes changing community
        nb_moves = 0
        # test each node
        for i in rand_nodes:
            # get the community of node i
            ci_old = C[i,0:C.shape[1]].toarray()[0]
            # modify C for our purposes, i.e., exclude node i
            C[i,0:C.shape[1]] = 0
            # build vector v for evaluating modularity increase
            v = (A[i]-d_out[0,i]*d_in-d_in[0,i]*d_out)/2
            # find the maximum - best community
            ci = soft_assign(v[0,i],np.array(v*C)[0])
            # update matrix
            C[i,0:C.shape[1]] = ci
            # update counter (if needed)
            nb_moves += np.linalg.norm(ci-ci_old)

        print(nb_moves)
        # exit if no improvement
        if (nb_moves<1e-10): break
        # otherwise: remove empty communities
        C = C[:,np.unique(scipy.sparse.find(C)[1])]
        # set improvement and reshuffle nodes for next try
        improvement = True
        shuffle(rand_nodes)

    # exit
    return C, improvement

