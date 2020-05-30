import pyscf
import pyqmc
import pyqmc.optimize_orthogonal
import pyscf.mcscf
import numpy as np
import h5py 

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

    wf, to_opt, freeze = pyqmc.default_msj(mol, mf, mc)

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

