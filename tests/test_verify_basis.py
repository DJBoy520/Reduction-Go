"""VerifyResult 和 verify_basis 的单元测试。

覆盖三种场景：
1. 不传 t_recon → equation_checked=False, passed=False
2. 传入正确的 t_recon → equation_checked=True, passed=True
3. 传入错误的 t_recon → passed=False

同时验证 VerifyResult 的 __bool__ 兼容性。
"""

import numpy as np
import pytest


class TestVerifyResult:
    """VerifyResult 数据类本身的测试。"""

    def test_passed_is_true(self):
        from src.domain.result import VerifyResult
        r = VerifyResult(valid_structure=True, equation_checked=True, passed=True)
        assert r.passed is True
        assert bool(r) is True
        assert r.error == ""

    def test_passed_is_false(self):
        from src.domain.result import VerifyResult
        r = VerifyResult(valid_structure=True, equation_checked=True, passed=False, error="boom")
        assert r.passed is False
        assert bool(r) is False
        assert r.error == "boom"

    def test_not_checked(self):
        from src.domain.result import VerifyResult
        r = VerifyResult(valid_structure=True, equation_checked=False, passed=False, error="no t_recon")
        assert r.equation_checked is False
        assert r.passed is False
        assert bool(r) is False

    def test_bool_compat(self):
        """验证 __bool__ 返回 self.passed，兼容旧代码 if verify_basis(...) 写法。"""
        from src.domain.result import VerifyResult
        r_true = VerifyResult(True, True, True)
        r_false = VerifyResult(True, True, False)
        if r_true:
            pass  # should not raise
        else:
            pytest.fail("VerifyResult(passed=True) should be truthy")
        if r_false:
            pytest.fail("VerifyResult(passed=False) should be falsy")


class TestVerifyBasisLatticeAttack:
    """测试 src/lattice_attack.py 的 verify_basis 函数。"""

    def _make_toy_data(self, k=2, l=2, n=5, q=8380417):
        from src.poly_math import mat_vec_mul, vec_add_mod
        np.random.seed(42)
        A = np.random.randint(0, 100, (k, l, n))
        s1 = np.random.randint(0, 3, (l, n))
        s2 = np.random.randint(0, 3, (k, n))
        t = vec_add_mod(mat_vec_mul(A, s1, q), s2, q) % q
        return A, s1, s2, t, q

    def test_correct_keys_pass(self):
        """正确密钥 → passed=True。"""
        from src.lattice_attack import verify_basis
        A, s1, s2, t, q = self._make_toy_data()
        result = verify_basis(A, t, q, s1, s2)

        assert result.equation_checked is True
        assert result.passed is True
        assert result.valid_structure is True

    def test_wrong_keys_fail(self):
        """错误密钥 → passed=False。"""
        from src.lattice_attack import verify_basis
        A, s1, s2, t, q = self._make_toy_data()
        wrong_s1 = np.zeros_like(s1)
        result = verify_basis(A, t, q, wrong_s1, s2)

        assert result.equation_checked is True
        assert result.passed is False
        assert "失败" in result.error

    def test_result_is_verify_result_type(self):
        """返回值必须是 VerifyResult 实例。"""
        from src.lattice_attack import verify_basis
        from src.domain.result import VerifyResult
        A, s1, s2, t, q = self._make_toy_data()
        result = verify_basis(A, t, q, s1, s2)

        assert isinstance(result, VerifyResult)
