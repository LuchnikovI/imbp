@testset "Test Eval By Tensor" begin
    @test [1, 2, 3] == IMBP.evolve_by_tensor_perm_order(3, 2, (3,), (1,))
    @test [1, 2, 3] == IMBP.evolve_by_tensor_perm_order(3, 2, (3,), (2,))
    @test [1, 3, 2] == IMBP.evolve_by_tensor_perm_order(3, 2, (2,), (1,))
    @test [1, 3, 2] == IMBP.evolve_by_tensor_perm_order(3, 2, (2,), (2,))
    @test [3, 1, 2] == IMBP.evolve_by_tensor_perm_order(3, 2, (1,), (1,))
    @test [3, 1, 2] == IMBP.evolve_by_tensor_perm_order(3, 2, (1,), (2,))
    @test [1, 6, 2, 3, 5, 4] == IMBP.evolve_by_tensor_perm_order(6, 4, (5, 2), (1, 4))
    @test [1, 6, 2, 3, 5, 4] == IMBP.evolve_by_tensor_perm_order(6, 4, (2, 5), (4, 1))
    @test [1, 7, 6, 2, 3, 4, 5] == IMBP.evolve_by_tensor_perm_order(7, 6, (3, 2, 7), (4, 6, 3))
    @test [1, 7, 6, 2, 3, 4, 5] == IMBP.evolve_by_tensor_perm_order(7, 6, (7, 2, 3), (3, 6, 4))
end