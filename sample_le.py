import pyscf
import pyqmc
import pyqmc.optimize_orthogonal, pyqmc.obdm
import pyscf.mcscf
import pyscf.lo
import numpy as np
import h5py 
import pyqmc.tbdm

def hf_casci(txt, chkfile='hf.chk', casfile='cas.chk', nelecas=(1,1), ncas = 2, nroots =4):
    mol = pyscf.gto.Mole(atom = txt, basis = 'cc-pvdz', unit='bohr')
    mol.build()
    mf = pyscf.scf.RHF(mol)
    mf.chkfile = chkfile
    mf.kernel()


    mycas = pyscf.mcscf.CASCI(mf,ncas,nelecas=nelecas)
    mycas.fcisolver.nroots=nroots
    mycas.kernel()
    with h5py.File(casfile,'w') as f:
        f['e_tot'] = mycas.e_tot
        f['ci'] = np.asarray(mycas.ci)
        f['nelecas']= nelecas
        f['ncas'] = ncas
        f['nroots']= nroots


def gen_wf(chkfile, casfile, root_weights):
    mol = pyscf.lib.chkfile.load_mol(chkfile)
    mol.output = None
    mol.stdout = None

    mf = pyscf.scf.RHF(mol)
    mf.__dict__.update(pyscf.scf.chkfile.load(chkfile, "scf"))

    with h5py.File(casfile, "r") as f:
        mc = pyscf.mcscf.CASCI(mf, ncas=int(f["ncas"][...]), nelecas=f["nelecas"][...])
        mc.ci = f["ci"][0, ...]
        mc.ci=0.0
        for root, weight in root_weights.items():
            mc.ci+=weight*f['ci'][root, ...]

    wf, to_opt, freeze = pyqmc.default_msj(mol, mf, mc, ion_cusp=True)

    return {
        "mol":mol,
        "mf":mf,
        "mc":mc,
        "freeze":freeze,
        "to_opt":to_opt,
        "wf":wf
    }

def restart_wf(chkfile, casfile, wf_file):
    mc_calc = gen_wf(chkfile, casfile, {0:1.})
    with h5py.File(wf_file, "r") as hdf:
        if "wf" in hdf.keys():
            grp = hdf["wf"]
            for k in grp.keys():
                mc_calc['wf'].parameters[k] = np.array(grp[k])
    return mc_calc

def linear(mc_calc,nconfig=1000,**kwargs):
    configs = pyqmc.initial_guess(mc_calc['mol'], nconfig)
    acc = pyqmc.gradient_generator(mc_calc['mol'], mc_calc['wf'], to_opt=mc_calc['to_opt'], freeze=mc_calc['freeze'])
    pyqmc.line_minimization(mc_calc['wf'], configs, acc,  **kwargs)


def optimize_excited(mc_anchors, mc_calc, nconfig = 1000, **kwargs):
    acc = pyqmc.gradient_generator(mc_calc['mol'], mc_calc['wf'], to_opt=mc_calc['to_opt'])
    wfs = [x['wf'] for x in mc_anchors]
    wfs.append(mc_calc['wf'])
    configs = pyqmc.initial_guess(mc_calc['mol'], nconfig)

    pyqmc.optimize_orthogonal.optimize_orthogonal(wfs,configs, acc, **kwargs)



def evaluate_big1rdm(mc_calc, output, nconfig =1000, **kwargs):
    mol = mc_calc['mol']
    a = pyscf.lo.orth_ao(mol, 'lowdin')
    obdm_up = pyqmc.obdm.OBDMAccumulator(mol=mol, orb_coeff=a, spin=0)
    obdm_down = pyqmc.obdm.OBDMAccumulator(mol=mol, orb_coeff=a, spin=1)

    configs = pyqmc.initial_guess(mol, 1000)
    pyqmc.vmc(mc_calc['wf'], configs, hdf_file=output,
                accumulators = {
                    'obdm_up':obdm_up,
                    'obdm_down':obdm_down
                },
                **kwargs
    )


def gen_basis(mol, mf, obdm, threshold = 1e-2):
    """From an obdm, use IAOs to generate a minimal atomic basis for a given state """
    n = obdm.shape[0]
    obdm*=n
    w,v = np.linalg.eig(obdm)
    keep = np.abs(w) > threshold
    a = pyscf.lo.orth_ao(mol, 'lowdin')
    basis = np.dot(a,v[:,keep]).real
    iao = pyscf.lo.iao.iao(mol, basis)
    iao = pyscf.lo.vec_lowdin(iao, mf.get_ovlp())
    return iao


def convert_basis(mol, mf, hdf_vmc, output, warmup=10):
    with h5py.File(hdf_vmc,'r') as vmc_hdf:
        obdm_up = np.mean(np.array(vmc_hdf['obdm_upvalue'][warmup:,...]),axis=0)
        obdm_down = np.mean(np.array(vmc_hdf['obdm_downvalue'][warmup:,...]),axis=0)
    basis_up = gen_basis(mol, mf, obdm_up)
    basis_down = gen_basis(mol, mf, obdm_down)
    with h5py.File(output, 'w') as f:
        f['basis_up'] = basis_up
        f['basis_down'] = basis_down


def evaluate_smallbasis(mc_calc, smallbasis_hdf, nconfig=1000, **kwargs):
    with h5py.File(smallbasis_hdf,'r') as f:
        basis_up = f['basis_up'][...]
        basis_down = f['basis_down'][...]
    mol = mc_calc['mol']
    obdm_up_acc = pyqmc.obdm.OBDMAccumulator(mol=mol, orb_coeff=basis_up, spin=0)
    obdm_down_acc = pyqmc.obdm.OBDMAccumulator(mol=mol, orb_coeff=basis_down, spin=1)
    tbdm = pyqmc.tbdm.TBDMAccumulator(mol, np.array([basis_up,basis_down]), spin=(0,1))
    acc = {'energy': pyqmc.EnergyAccumulator(mol),
        'obdm_up':obdm_up_acc,
        'obdm_down':obdm_down_acc,
        'tbdm': tbdm } 

    configs = pyqmc.initial_guess(mc_calc['mol'], nconfig)
    pyqmc.vmc(mc_calc['wf'], configs, accumulators = acc, **kwargs)
