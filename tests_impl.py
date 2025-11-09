import numpy as np
import autograd.numpy as anp
from autograd import grad, hessian
from scipy.optimize import minimize
from scipy.stats import chi2
from typing import Callable, Tuple, List, Dict, Any


def vec(A):
    return anp.reshape(A.T, (-1,))


def make_selector_P(m: int, k: int, f2_indexes: List[int]) -> anp.ndarray:
    """
    P: матрица (m2*k) x (m*k), такая что vec(G^T) -> vec(G2^T).
    Порядок vec(G^T): блоки длины k по моментам i=0..m-1.
    """
    f2_indexes = list(sorted(f2_indexes))
    m2 = len(f2_indexes)
    P = np.zeros((m2 * k, m * k))
    for r, i in enumerate(f2_indexes):
        P[r*k:(r+1)*k, i*k:(i+1)*k] = np.eye(k)
    return anp.asarray(P)


def unconditional_relevance(
    data: Dict[str, np.ndarray],
    moments: List[Callable],
    f2_indexes: List[int],
    theta_init: np.ndarray,
    ridge: float = 1e-8,
    verbose: bool = False
) -> Tuple[float, float, np.array]:
    '''
    Пайплайн теста на безусловную релевантность моментов f2.

    Params:
    ----------
    data: dict[str, np.array]
        Словарь с данным, где ключ - название переменной, значение - массив наблюдений
    f: list[Callable]
        Список моментных условий [f1, f2, ..., fm]
    f2_indexes: list[int]
        Список индексов моментных условий, подвергающихся тесту
    theta_init: np.array
        Начальные значения параметров
    ridge: float
        Параметр регуляризация для избежания сингулярности матрицы при инвертировании
    verbose: bool
        Показывать отладку работы теста (по-умолчанию отключено)

    Return:
    ----------
    tuple[float, float, np.array]
        Результаты теста: значение статистики, p-value, оценку theta
    '''
    def vprint(*args, **kwargs):
        '''
        Для отладки
        '''
        if verbose:
            print(*args, **kwargs)

    T = len(next(iter(data.values())))
    m = len(moments)
    theta_init = anp.asarray(theta_init)
    k = int(theta_init.size)
    f2_indexes = list(sorted(f2_indexes))
    m2 = len(f2_indexes)
    m1 = m - m2
    assert m2 > 0, "Нужно указать хотя бы один момент в f2_indexes"
    assert m1 >= k, "Требуется m1 >= k (идентифицируемость на f1)"

    data_ag = {key: anp.asarray(val) for key, val in data.items()}
    def get_data_point(t: int) -> Dict[str, anp.ndarray]:
        return {k: data_ag[k][t] for k in data_ag}

    def fbar(theta: anp.ndarray) -> anp.ndarray:
        rows = []
        for t in range(T):
            dp = get_data_point(t)
            rows.append(anp.array([mom(theta, dp) for mom in moments]))
        f_mat = anp.stack(rows)
        return anp.mean(f_mat, axis=0)

    def compute_f_and_G_all(theta: anp.ndarray):
        f_rows = []
        G_rows = []
        for t in range(T):
            dp = get_data_point(t)
            f_row = []
            G_row = []
            for mom in moments:
                val = mom(theta, dp)
                f_row.append(val)
                g = grad(lambda p: mom(p, dp))(theta)
                G_row.append(g)
            f_rows.append(anp.stack(f_row))
            G_rows.append(anp.stack(G_row))        
        f_mat = anp.stack(f_rows)   
        G_tens = anp.stack(G_rows)               
        G = anp.mean(G_tens, axis=0)  
        return f_mat, G_tens, G

    def compute_Omega_from_f_mat(f_mat: anp.ndarray) -> anp.ndarray:
        f_mean = anp.mean(f_mat, axis=0)
        f_c = f_mat - f_mean
        return (f_c.T @ f_c) / (T - 1)

    # 1) старт: Omega = I
    def gmm_objective_init(theta):
        fb = fbar(anp.asarray(theta))
        return float(fb.T @ fb)

    vprint('[ESTIMATING INITIAL THETA]: ', end='')
    res1 = minimize(
        lambda th: gmm_objective_init(th),
        np.array(theta_init, dtype=np.float64),
        method='BFGS'
    )
    theta0 = anp.asarray(res1.x)
    vprint('DONE')

    # 2) оптимальная стадия: Omega != I (more complex)
    vprint('[ESTIMATING OMEGA]: ', end='')
    f_mat0, G_tens0, G0 = compute_f_and_G_all(theta0)
    Omega0 = compute_Omega_from_f_mat(f_mat0)
    vprint('DONE')

    def gmm_objective(theta):
        fb = fbar(anp.asarray(theta))
        Oinv = anp.linalg.inv(Omega0 + ridge * anp.eye(m))
        return float(fb.T @ Oinv @ fb)

    vprint('[ESTIMATING FINAL THETA]: ', end='')
    res2 = minimize(
        lambda th: gmm_objective(th),
        np.array(theta0, dtype=np.float64),
        method='BFGS'
    )
    theta_hat = anp.asarray(res2.x)
    vprint("DONE")

    # 3) оценки элементов из r (после окончательной оценки theta)
    vprint('[ESTIMATING FINAL OMEGA]: ', end='')
    f_mat, G_tens, G = compute_f_and_G_all(theta_hat)
    Omega = compute_Omega_from_f_mat(f_mat)
    Oinv = anp.linalg.inv(Omega + ridge * anp.eye(m))
    vprint("DONE")

    # 4) g2 = vec(G2^T)
    G2 = G[f2_indexes, :]
    g2 = vec(G2.T)

    # 5) H2 — блок-диагональ из Hessian(mean f2_i)
    vprint('[ESTIMATING H2]: ', end='')
    def phi_i_factory(i: int):
        def phi_i(theta):
            s = 0.0
            for t in range(T):
                s = s + moments[i](theta, get_data_point(t))
            return s / T
        return phi_i

    H2 = anp.zeros((m2 * k, k))
    for r, i in enumerate(f2_indexes):
        Hi = hessian(phi_i_factory(i))(theta_hat)
        H2 = anp.concatenate([H2[:r*k, :],
                              Hi,
                              H2[(r+1)*k:, :]], axis=0) if r < m2 else Hi
    vprint("DONE")

    # 6) B = [B_f, P]
    vprint('[ESTIMATING B]: ', end='')
    middle = G.T @ Oinv @ G
    middle_inv = anp.linalg.inv(middle + ridge * anp.eye(k))
    B_f = H2 @ middle_inv @ G.T @ Oinv
    P = make_selector_P(m=m, k=k, f2_indexes=f2_indexes)
    B = anp.concatenate([B_f, P], axis=1)
    vprint("DONE")

    vprint('[ESTIMATING SIGMA R]: ', end='')
    R_rows = []
    for t in range(T):
        ft = f_mat[t, :]
        gt = vec(G_tens[t, :, :].T)
        R_rows.append(anp.concatenate([ft, gt]))
    R = anp.stack(R_rows)

    R_mean = anp.mean(R, axis=0)
    Rc = R - R_mean
    Sigma_r = (Rc.T @ Rc) / (T - 1)
    vprint("DONE")

    Sigma_g2 = B @ Sigma_r @ B.T
    Sigma_g2 = Sigma_g2 + ridge * anp.eye(m2 * k)
    W = float(T * (g2.T @ anp.linalg.inv(Sigma_g2) @ g2))
    p_value = 1.0 - chi2.cdf(W, df=m2 * k)
    vprint("[TEST DONE]")

    return W, float(p_value), theta_hat