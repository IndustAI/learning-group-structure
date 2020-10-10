import torch

class Representation():

    def __init__(self, dim=4):
        self.dim = dim
        self.params = dim * (dim - 1) // 2
        self.thetas = torch.autograd.Variable((2*torch.rand(self.params)-1) / dim, requires_grad=True)

        self.__matrix = None

    def set_thetas(self, thetas):
        self.thetas = thetas
        self.thetas.requires_grad = True
        self.clear_matrix()

    def clear_matrix(self):
        self.__matrix = None

    def get_matrix(self):
        if self.__matrix is None:
            k = 0
            mats = []
            for i in range(self.dim - 1):
                for j in range(self.dim - 1 - i):
                    theta_ij = self.thetas[k]
                    k += 1
                    c, s = torch.cos(theta_ij), torch.sin(theta_ij)

                    rotation_i = torch.eye(self.dim, self.dim)
                    rotation_i[i, i] = c
                    rotation_i[i, i + j + 1] = s
                    rotation_i[j + i + 1, i] = -s
                    rotation_i[j + i + 1, j + i + 1] = c

                    mats.append(rotation_i)

            def chain_mult(l):
                if len(l) >= 3:
                    return l[0] @ l[1] @ chain_mult(l[2:])
                elif len(l) == 2:
                    return l[0] @ l[1]
                else:
                    return l[0]

            self.__matrix = chain_mult(mats)

        return self.__matrix