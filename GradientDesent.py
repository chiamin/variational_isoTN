import copy, sys
sys.path.insert(0,'/home/chiamin/Cytnx/Install/')
import cytnx
import Utility as ut
import scipy as sp
import UniTensorTools as uniten
import numpy as np
import math
import matplotlib.pyplot as plt

# cost_func must have two functions: value(x) and gradient(x)
def gradient_descent (cost_func, x, step_size, N_linesearch=10, c1=1e-4, c2=0.9, **args):
    grad = cost_func.gradient(x)

    direction = -grad

    phi = LineSearchFunction(cost_func, x, direction, **args)

    x_new, value, slope = LineSearch(phi, step_size, c1=c1, c2=c2, nIter=N_linesearch)
    return x_new, value, slope

# Perform x + a*d with constraint on x
def walk_x (x, d, a, **args):
    if a == 0:
        return x

    if "constraint" not in args:
        return x + d*a

    elif args["constraint"] == "np normalize":
        x_new = x + d*a
        x_new = x_new / np.linalg.norm(x_new)
        return x_new

    elif args["constraint"] in ["np AL", "np AR"]: # ALNp and ARNp mean AL and AR as Numpy.array
        s = x.shape
        if args["constraint"] == 'np AL':
            assert x.ndim == 3
            x = x.reshape((s[0]*s[1],s[2]))
            d = d.reshape((s[0]*s[1],s[2]))
        elif args["constraint"] == 'np AR':
            assert x.ndim == 3
            x = x.reshape((s[0],s[1]*s[2])).transpose()
            d = d.reshape((s[0],s[1]*s[2])).transpose()
        #assert ut.is_isometry(x)

        xd = ut.outer_dot(x, d)
        dx = xd.T.conj()
        Q = xd - dx
        expQ = sp.linalg.expm(-a*Q)
        x_new = expQ @ x
        #assert ut.is_isometry(x_new)

        if args["constraint"] == 'np AR':
            x_new = x_new.transpose()
        return x_new.reshape(s)

    elif args["constraint"] == "UniTen isometry":    # UniTensor
        assert x.labels() == d.labels()
        #           s
        #           |
        #           |
        #  l -----(x,d)----- r

        # row_labels contains the labels for the column indices
        row_labels = args["row_labels"]
        for rlabel in row_labels:
            assert rlabel in x.labels()
        # "Prime" the row labels
        rowp_labels = [i+"'" for i in row_labels]

        # r1, r2, r3 are row labels
        # c1, c2 are column labels
        #           ___        ___
        #   r1 ----|   |  c1  |   |---- r1'
        #          |   |------|   |
        #   r2 ----| x |  c2  | d'|---- r2'
        #          |   |------|   |
        #   r3 ----|___|      |___|---- r3'
        #
        xd = cytnx.Contract(x, d.Conj().relabels(row_labels, rowp_labels))
        dx = cytnx.Contract(d, x.Conj().relabels(row_labels, rowp_labels))
        #print('-------------------------------------------')
        #print(x.labels(), d.labels(), d.relabels(row_labels, rowp_labels).labels())
        #print(x.labels(), d.labels(), x.relabels(row_labels, rowp_labels).labels())
        #print(xd.labels(), dx.labels())
        assert xd.labels() == dx.labels()

        Q = xd - dx
        Q.relabels_(xd.labels())

        # Make Q a square matrix
        # 1. Permute Q such that row and column indices have the same order
        Q.permute_(row_labels+rowp_labels)
        # 2. Set rowrank to the number of row indices
        Q.set_rowrank_(len(row_labels))

        try:
            expQ = cytnx.linalg.ExpM(Q, -a)
            expQ = uniten.ToReal(expQ)
        except RuntimeError:
            tmp = uniten.ToNpArray(Q)
            print(tmp)
            shape = tmp.shape
            dim = math.isqrt(tmp.size)
            tmp = tmp.reshape((dim,dim))
            tmp = sp.linalg.expm(-a*tmp)
            print(tmp)
            tmp = tmp.reshape(shape)
            expQ = uniten.ToUniTensor(tmp, Q.labels())
            #print(Q.shape(), Q.labels(), expQ.labels())
            raise Exception
        #           ________   r1'  ___
        #   r1 ----|        |-------|   |
        #          |        |  r2'  |   |---- c1
        #   r2 ----| exp(Q) |-------| x |
        #          |        |  r3'  |   |---- c2
        #   r3 ----|________|-------|___|
        #
        x_new = cytnx.Contract(expQ, x.relabels(row_labels, rowp_labels))
        x_new.permute_(x.labels())

        #print('*',x.dtype_str(), expQ.dtype_str(), x_new.dtype_str())
        #print(x_new)
        #gg = expQ.astype(cytnx.Type.Double)
        #print(expQ.dtype_str())

        #exit()
        return x_new

    elif args["constraint"] == "UniTen normalize":
        x_new = x + d*a
        x_new = x_new / x_new.Norm().item()
        x_new.relabels_(x.labels())
        return x_new

    elif args["constraint"] == "UniTen":
        x_new = x + d*a
        x_new.relabels_(x.labels())
        return x_new

    else:
        print("Unkown type:",self.type)
        raise Exception

class LineSearchFunction:
    def __init__ (self, cost_func, x0, direction, **args):
        self.func = cost_func
        self.x0 = x0
        self.d = direction
        self.args = args

    def value_slope (self, a, da=1e-6):
        x1 = walk_x(self.x0, self.d, a, **self.args)
        x2 = walk_x(self.x0, self.d, a+da, **self.args)
        f1 = self.func.value(x1)
        f2 = self.func.value(x2)
        slope = (f2 - f1)/da
        return f1, slope, x1

def sufficient_decrease_condition (f, f0, df0, c1, a):
    return f <= f0 + c1*a*df0

def curvature_condition (df, df0, c2):
    return abs(df) <= abs(c2*df0)

def strong_Wolfe_condition (f, df, f0, df0, c1, c2, a):
    cond1 = sufficient_decrease_condition (f, f0, df0, c1, a)
    cond2 = curvature_condition (df, df0, c2)
    return cond1 and cond2

def LineSearch (func, step_size, c1=1e-4, c2=0.9, nIter=10, min_step=1e-12, debug=False):
    f0, df0, x0 = func.value_slope(0)
    f_pre, df_pre = f0, df0

    a_pre = 0
    a = step_size
    first_iter = True

    for c in range(nIter):
        assert a_pre < a

        f, df, x = func.value_slope (a)

        if debug:
            plot_func_vs_step (func, a_pre, a, c1=c1, c2=c2)
            plt.show()

        # sufficient_decrease_condition is not satisfied
        if not sufficient_decrease_condition (f, f0, df0, c1, a) or (f >= f_pre and not first_iter):
            if debug:
                print('violate sufficient decrease condition; search between', a_pre, a)

            # search in the window
            a = search_interval (func, a_pre, a, f_pre, f, df_pre, df)
            return x, f, df

        # Both sufficient_decrease_condition and curvature_condition are satisfied
        elif curvature_condition (df, df0, c2):
            if debug:
                print('get a solution')

            return x, f, df

        # sufficient_decrease_condition is satisfied
        # curvature_condition is not satisfied
        # curvature is positive
        elif df >= 0:
            if debug:
                print('positive slope, search between', a, a_pre)

            # search in the window
            x, f, df = search_interval (func, a, a_pre, f, f_pre, df, df_pre)
            return x, f, df

        # sufficient_decrease_condition is satisfied
        # curvature_condition is not satisfied
        # curvature is negative
        # --> the whole window does not satisfy Wolfe condition
        else:
            # enlarge the windown
            step_size *= 2
            a_pre, f_pre, df_pre = a, f, df
            a = a + step_size
            if debug:
                print('extended the window')

        first_iter = False
    return x, f, df

def search_interval (func, a_lo, a_hi, f_lo, f_hi, df_lo, df_hi, c1=1e-4, c2=0.9, nIter=10, debug=False):
    def assert_condition (f_lo, f_hi, df_lo, a_hi, a_lo):
        assert f_lo <= f_hi
        #assert df_lo*(a_hi-a_lo) < 0
    #assert_condition (f_lo, f_hi, df_lo, a_hi, a_lo)

    a_lo0, a_hi0 = a_lo, a_hi
    def plot (a_lo, a_hi, f_lo, f_hi, a):
        plot_func_vs_step (func, a_lo0, a_hi0, c1=c1, c2=c2)
        ax = plt.gca()
        ax.plot ([a_lo], [f_lo], c='r', marker='v', ms=10)
        ax.plot ([a_hi], [f_hi], c='r', marker='^', ms=10)
        ax.plot ([a], [f], c='r', marker='o', ms=10)
        ax.plot ([a], [df], c='r', marker='x', ms=10)
        return ax

    f0, df0, x0 = func.value_slope(0)
    for c in range(nIter):
        #assert_condition (f_lo, f_hi, df_lo, a_hi, a_lo)
        if abs(a_lo-a_hi) < 1e-12:
            print('Step is less than 1e-12')
            return x0, f0, df0

        # bisection search
        a = 0.5 * (a_lo + a_hi)
        f, df, x = func.value_slope (a)

        # sufficient_decrease_condition is not satisfied
        if not sufficient_decrease_condition (f, f0, df0, c1, a) or f >= f_lo:
            # move to the a_lo window
            a_hi = a
            f_hi = f
            df_hi = df
            #assert_condition (f_lo, f_hi, df_lo, a_hi, a_lo)

            if debug:
                ax = plot (a_lo, a_hi, f_lo, f_hi, a)
                ax.axvspan(a_lo, a_hi, alpha=0.2, color='gray')
                print('11')
                plt.show()
        else:
            # sufficient_decrease_condition is satisfied
            # curvature_condition is satisfied
            if curvature_condition (df, df0, c2):
                if debug:
                    ax = plot (a_lo, a_hi, f_lo, f_hi, a)
                    ax.plot ([a], [f], c='r', marker='*', ms=20)
                    print('22')
                    plt.show()
                return x, f, df
            # sufficient_decrease_condition is satisfied
            # curvature_condition is not satisfied
            # df * (a_hi - a_lo) >= 0
            elif df * (a_hi - a_lo) >= 0:
                if debug:
                    print('33')

                # switch a_lo and a_hi
                a_hi = a_lo
                f_hi = f_lo
                df_hi = df_lo
            a_lo = a
            f_lo = f
            df_lo = df

            if debug:
                ax = plot (a_lo, a_hi, f_lo, f_hi, a)
                ax.axvspan(a_lo, a_hi, alpha=0.2, color='gray')
                plt.show()
    return x, f, df











'''def LineSearch (func, step_size, c1=1e-4, c2=0.9, nIter=10, min_step=1e-12, debug=True):
    a0 = 0
    a = step_size

    f0, df0, x0 = func.value_slope(0)
    fa, dfa, xa = func.value_slope(a)

    if df0 > 0.:
        print('Positive initial slope:',df0)
        plot_func_vs_step (func, a0, a, c1, c2)
        plt.show()
        raise Exception

    for i in range(nIter):

        if debug:
            plot_func_vs_step (func, a0, a, c1, c2)
            plt.show()

        if not sufficient_decrease_condition(fa, f0, df0, c1, a-a0):
            # Search the left window
            a = 0.5*(a0 + a)
            fa, dfa, xa = func.value_slope(a)

            if debug:
                print("violate sufficient decrease condition; search the right window")
        else:
            if curvature_condition (dfa, df0, c2):
                if debug:
                    print("get solution")
                return xa, fa, dfa
            elif dfa < 0.:
                # Extend the window
                step_size *= 2
                a = a + step_size
                fa, dfa, xa = func.value_slope(a)
                if debug:
                    print("extend the window")
            else:
                # Search the right window
                a0 = 0.5*(a0 + a)
                f0, df0, x0 = func.value_slope(a0)
                if debug:
                    print("positive slope, search the right window")

        assert a0 < a
        # If the step size smaller than min_step, return the current step
        if a - a0 < min_step:
            break

    return xa, fa, dfa'''

def plot_func_vs_step (func, a1, a2, c1=0, c2=0, Na=200):
    f0, df0, x0 = func.value_slope(0)

    fs,dfs,dfs2,fs_Wolfe = [],[],[],[]
    fpre = 0
    a_scan = np.linspace(a1,a2,Na)
    da = a_scan[1] - a_scan[0]
    for a in a_scan:
        f,df,x = func.value_slope (a)
        fs.append(f)
        dfs.append(df)
        dfs2.append((f-fpre)/da)
        fpre = f
        if c1 != 0 and c2 != 0:
            if strong_Wolfe_condition (f, df, f0, df0, c1, c2, a):
                fs_Wolfe.append(f)
            else:
                fs_Wolfe.append(None)
    dfs2[0] = float('Nan')
    plt.plot(a_scan,fs,marker='.',c='tab:blue',label='f')
    plt.plot(a_scan,dfs,marker='x',c='tab:orange',label='df')
    plt.plot(a_scan-0.5*da,dfs2,marker='+',c='tab:green',label='df2')
    if c2 != 0:
        plt.axhline(c2*df0,ls='--',c='gray')
        plt.axhline(-c2*df0,ls='--',c='gray')
    if len(fs_Wolfe) != 0:
        plt.plot(a_scan,fs_Wolfe,marker='o',c='darkblue',label='Wolfe')
    plt.legend()

