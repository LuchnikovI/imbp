@testset "Exact Simulator Testing" begin
    # ghz state test
    qs = QSim(Float64, 10)
    hadamard = (1 / sqrt(2)) * ComplexF64[1 1 ; 1 -1]
    cnot = ComplexF64[1 0 0 0 ; 0 1 0 0 ; 0 0 0 1 ; 0 0 1 0]
    apply_one_qubit_gate!(qs, hadamard, 2)
    apply_two_qubit_gate!(qs, cnot, 2, 1)
    apply_two_qubit_gate!(qs, cnot, 1, 3)
    apply_two_qubit_gate!(qs, cnot, 1, 4)
    apply_two_qubit_gate!(qs, cnot, 4, 5)
    apply_two_qubit_gate!(qs, cnot, 5, 7)
    apply_two_qubit_gate!(qs, cnot, 7, 6)
    apply_two_qubit_gate!(qs, cnot, 6, 8)
    apply_two_qubit_gate!(qs, cnot, 8, 9)
    apply_two_qubit_gate!(qs, cnot, 8, 10)
    for pos in 1:10
        @test norm(get_one_qubit_dens(qs, pos) - ComplexF64[0.5 0. ; 0. 0.5]) < 1e-10
    end
end