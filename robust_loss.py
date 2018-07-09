import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class criterion_diag( nn.Module ):
    def __init__( self, ncls, ndf, fc, alp=0.1, moving_avg = False ):
        super().__init__()
        self.register_buffer( 'mean', torch.zeros((ncls, ndf)) )
        self.register_buffer( 'var', torch.zeros((ncls, ndf))  )
        self.it = [0 for i in range(ncls)]
        self.ncls, self.ndf, self.alp = ncls, ndf, alp
        self.moving_avg = moving_avg
        self.fc = fc

    def forward( self, df, gt ):
        alp = self.alp
        ndf, ncls = self.ndf, self.ncls
        fc = self.fc
        with torch.no_grad():
            for i in range(df.size(0)):
                f = df[i].detach()
                t = gt[i]
                it = self.it[t]
                self.it[t] += 1
                if self.moving_avg:
                    c1, c2 = 0.999, 0.001
                else:
                    c1, c2 = (it / (it + 1)), (1 / (it + 1))
                if it == 0:
                    self.mean[t] = f
                else:
                    self.mean[t] = self.mean[t] * c1 + f * c2
                a = (f - self.mean[t])**2
                if self.moving_avg:
                    self.var[t] = self.var[t] * c1 + a * c2
                else:
                    self.var[t] = self.var[t] * c1 + a * c1 * c2
        fc = fc.transpose( 0, 1 )
        # (minibatch_size * ndf)
        Var = self.var[ gt ]
        # (minibatch_size * ndf)
        Fc = fc[:, gt].transpose(0, 1)
        fc = fc.transpose(0, 1)
        size = (gt.size(0), ncls, ndf)
        # (minibatch_size, ndf, ncls)
        W = fc[None, :, :].expand( size ) - Fc[:, None, :].expand( size )

        #W = W.transpose( 1, 2 )
        df = df.view( df.size(0), ndf, 1 )
        Var = Var.view( Var.size(0), ndf, 1 )
        
        score = torch.matmul( W, df ) + alp * 0.5 * torch.matmul( W**2, Var )

        #loss = score.exp().sum(1).log().mean()
        """
        z_k, _ = score.max(1)
        score = score - z_k[:, None, :].expand( score.size() )
        loss = ((score.exp().sum(1)+1e-6).log() + z_k).mean()
        """
        return score

class criterion_mat( nn.Module ):
    def __init__( self, ncls, ndf, fc, alp=0.1, moving_avg=False, indepen=False ):
        super().__init__()
        self.mean = torch.zeros( (ncls, ndf) ).cuda()
        self.cov_mat = torch.zeros( (ncls, ndf, ndf )).cuda()
        self.it = [0 for i in range(ncls)]
        self.ncls = ncls
        self.ndf = ndf
        self.alp = alp
        self.moving_avg = moving_avg
        self.indepen = indepen
        self.fc = fc

    def forward( self, df, gt ):
        loss = 0
        fc = self.fc
        alp = self.alp
        score = []
        for i in range(df.size(0)):
            f = df[i].detach() 
            t = gt[i]
            it = self.it[t]
            self.it[t] += 1
            c1, c2 = (it / (it + 1)), (1 / (it + 1))
            if self.moving_avg:
                c1, c2 = 0.999, 0.001
            if it == 0:
                self.mean[t] = f
            else:
                self.mean[t] = self.mean[t] * c1 + f * c2
            a = f - self.mean[t]
            mat = torch.matmul(a.view(self.ndf, 1), a.view(1, self.ndf))
            if not self.moving_avg:
                self.cov_mat[t] = self.cov_mat[t] * c1 + mat * c1 * c2
            else:
                self.cov_mat[t] = self.cov_mat[t] * c1 + mat * c2
            if self.indepen:
                self.cov_mat[t] = self.cov_mat[t] * torch.eye( self.cov_mat[t].size(0) ).cuda()
            mat_v = Variable( torch.from_numpy(self.cov_mat[t].data.cpu().numpy()) ).cuda()
            mat_v = mat_v.type( torch.cuda.FloatTensor )

            f = df[i]
            fc1 = fc - fc[t]
            if it < 000:
                loss_a = torch.matmul( fc1, f ).exp().sum().log()
            else:
                z = torch.matmul( fc1, f ) + 1/2 * alp * torch.matmul( fc1, torch.matmul(mat_v, fc1.t())).diag()
                score.append( z[None, :] )
                #z_k = z.max()
                #z = z - z_k
                #loss_a = z_k + (z.exp().sum()+1e-5).log()
                #print((torch.matmul( fc1, f ), 1/2 * torch.matmul( fc1, torch.matmul(mat_v, fc1.t())).diag()))
            """
            loss_f = loss_a.data[0]
            if np.isnan(loss_f) or loss_f > 1e+5:
                pass
            else:
                loss += loss_a
            """
        score = torch.cat( score, 0 )
        score = score[:, :, None]
        return score
