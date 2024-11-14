using ITensors
using PyPlot

# 3. Define the Hamiltonian using AutoMPO
function add_h(ampo, h, Nx, Ny)
    total_sites = Nx * Ny
    for i in 1:total_sites
        add!(ampo, h, "Sz", i)
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
                add!(ampo, J, "Sx", i, "Sx", j)
                Nbond += 1
            else
                j = site_index(1, y, Ny)
                add!(ampo, J, "Sx", i, "Sx", j)
                Nbond += 1
            end

            # Vertical coupling (σ^z_i σ^z_{i+Nx}) if within bounds
            if y < Ny
                j = site_index(x, y+1, Ny)
                add!(ampo, J, "Sx", i, "Sx", j)
                Nbond += 1
            end
        end
    end
    return ampo, Nbond
end

function bond_H(J, h, x1, y1, x2, y2, Nx, Ny)
    @assert x1 <= Nx && x2 <= Nx && y1 <= Ny && y2 <= Ny
    ampo = AutoMPO()
    i = site_index(x1, y1, Ny)
    j = site_index(x2, y2, Ny)
    add!(ampo, J, "Sx", i, "Sx", j)
    add!(ampo, h/3, "Sz", i)
    add!(ampo, h/3, "Sz", j)
    return ampo
end

function runDMRG(Nx, Ny, h, J)
    # 2. Set up the sites for a spin-1/2 system
    total_sites = Nx * Ny
    sites = siteinds("S=1/2", total_sites)

    ampo = AutoMPO()
    ampo = add_h(ampo, h, Nx, Ny)
    ampo, Nbond = add_J(ampo, J, Nx, Ny)
    # Convert the Hamiltonian to an MPO
    H = MPO(ampo, sites)

    ampo = AutoMPO()
    ampo = add_h(ampo, h, Nx, Ny)
    Hh = MPO(ampo, sites)
    ampo = AutoMPO()
    ampo, Nbond = add_J(ampo, J, Nx, Ny)
    HJ = MPO(ampo, sites)

    # 4. Set up and run DMRG
    sweeps = Sweeps(20)
    maxdim!(sweeps, 20, 40, 80, 100)
    cutoff!(sweeps, 1e-10)

    # Initialize a random MPS
    psi0 = randomMPS(sites, 10)

    # Run DMRG to find the ground state
    energy, psi = dmrg(H, psi0, sweeps)

    en_h = inner(psi',Hh,psi)/total_sites
    en_J = inner(psi',HJ,psi)/Nbond
    #println("E(h): ",en_h)
    #println("E(J): ",en_J)
    Hx1 = MPO(bond_H(J, h, div(Nx,2), 1, div(Nx,2)+1, 1, Nx, Ny), sites)
    Hx2 = MPO(bond_H(J, h, div(Nx,2), 2, div(Nx,2)+1, 2, Nx, Ny), sites)
    Hy1 = MPO(bond_H(J, h, div(Nx,2), 1, div(Nx,2), 2, Nx, Ny), sites)
    enx1 = inner(psi',Hx1,psi)
    enx2 = inner(psi',Hx2,psi)
    eny1 = inner(psi',Hy1,psi)

    println("E ",energy/total_sites," ",enx1," ",enx2," ",eny1)
    return en_h/h
end

function main()
    # 1. Define parameters
    Nx, Ny = 200, 2 # Lattice dimensions (4x4 grid for example)
    h = -1.0        # Transverse field strength
    J = -1.0        # Coupling strength

    sz = runDMRG(Nx, Ny, h, J)


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
