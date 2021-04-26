class KNN: #con numpy
    def __init__(self):
        pass

    def fit(self,x,y):
        self.y = y
        self.x = x
        
    def predict(self,x,y=None, k =3, test=False,**kwargs):
      
        distances = []
        dim = x.shape[1]
       
        dist = np.sqrt(np.sum((self.x-x)**2,axis=-1))
        sdist = dist.argsort()
        neighbors = self.y[sdist]
        if not test:
            preds = list(neighbors[:k])

            preds = {i[0]:preds.count(i)/len(preds) for i in preds}
            return preds
        preds = [x[0] for x in neighbors[:k]]
       
        return max(set(preds), key=preds.count)

    def predict_minkowski(self,x,y=None,  k =3,test = False,**kwargs):
        p = kwargs.get("p",2)
        distances = []
        dim = x.shape[1]
       
        dist = np.abs(np.sum((self.x-x)**p,axis=-1))**(1/p)
        sdist = dist.argsort()
        neighbors = self.y[sdist]
        if not test :
            preds = list(neighbors[:k])
            preds = {i[0]:preds.count(i)/len(preds) for i in preds}
            return preds
        preds = [x[0] for x in neighbors[:k]]
       
        return max(set(preds), key=preds.count)
    
    
    def predict_manhattan(self,x,y=None, k =3,test=False,**kwargs):
        distances = []
        dim = x.shape[1]
       
        dist = (np.sum(np.abs(self.x-x),axis=-1))
        sdist = dist.argsort()
        neighbors = self.y[sdist]
        if not test:
            preds = list(neighbors[:k])
            preds = {i[0]:preds.count(i)/len(preds) for i in preds}
            return preds
        preds = [x[0] for x in neighbors[:k]]
       
        return max(set(preds), key=preds.count)
    
    
    
    def test(self,xs,y,mode="euclid",k=3,**kwargs):
        d = {"euclid":self.predict,
            "minkowski":self.predict_minkowski,
            "manhattan":self.predict_manhattan}
        acc = []
        for i,x in enumerate(xs):
    
            preds = d[mode](np.array([x]),k=k,test=True,)
            
            if preds==y[i]:
                acc.append(1)
            else:
                acc.append(0)
        return sum(acc)/len(acc)
        
        
        
      
        
        
        
knn = KNN()
