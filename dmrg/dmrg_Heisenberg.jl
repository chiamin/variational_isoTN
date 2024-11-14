using ITensors
using PyPlot

# 3. Define the Hamiltonian using AutoMPO
function totalSz(ampo, Nx, Ny)
    total_sites = Nx * Ny
    for i in 1:total_sites
        add!(ampo, "Sz", i)
    end
    return ampo
end

# Coupling term (σ^z σ^z) for nearest neighbors in the 2D grid
function site_index(x, y, Ny)
    return (x - 1) * Ny + y
end

function add_J(ampo, J, Nx, Ny)
    Nbond = 0
    for x in 1:Nx
        for y in 1:Ny
            i = site_index(x, y, Ny)

            # Horizontal coupling (σ^z_i σ^z_{i+1}) if within bounds
            if x < Nx
                j = site_index(x+1, y, Ny)
                add!(ampo, J, "Sz", i, "Sz", j)
                add!(ampo, 0.5*J, "S+", i, "S-", j)
                add!(ampo, 0.5*J, "S+", j, "S-", i)
                Nbond += 1
            end
            # Vertical coupling (σ^z_i σ^z_{i+Nx}) if within bounds
            if y < Ny
                j = site_index(x, y+1, Ny)
                add!(ampo, J, "Sz", i, "Sz", j)
                add!(ampo, 0.5*J, "S+", i, "S-", j)
                add!(ampo, 0.5*J, "S+", j, "S-", i)
                Nbond += 1
            end
        end
    end
    return ampo, Nbond
end

function runDMRG(Nx, Ny, J)
    # 2. Set up the sites for a spin-1/2 system
    total_sites = Nx * Ny
    sites = siteinds("S=1/2", total_sites, conserve_qns=true)

    ampo = AutoMPO()
    ampo, Nbond = add_J(ampo, J, Nx, Ny)
    # Convert the Hamiltonian to an MPO
    H = MPO(ampo, sites)

    ampo = AutoMPO()
    ampo = totalSz(ampo, Nx, Ny)
    totSz = MPO(ampo, sites)

    # 4. Set up and run DMRG
    sweeps = Sweeps(20)
    maxdim!(sweeps, 20, 40, 80, 100)
    cutoff!(sweeps, 1e-10)

    # Initialize a random MPS
    psi0 = randomMPS(sites, 10)

    # Run DMRG to find the ground state
    energy, psi = dmrg(H, psi0, sweeps)
    energy /= total_sites

    sz = inner(psi',totSz,psi)/total_sites
    println("E ",energy," ",sz)
    return energy, sz
end

function main()
    # 1. Define parameters
    Nx, Ny = 100, 3 # Lattice dimensions (4x4 grid for example)
    J = -1.0        # Coupling strength

    runDMRG(Nx, Ny, J)

    #=hs = 0:0.2:3
    szs = []
    for h=hs
        sz = runDMRG(Nx, Ny, h, J)
        push!(szs, sz)
    end

    PyPlot.plot(hs, szs, marker=".")
    show()=#
end

main()
