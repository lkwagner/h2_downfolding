import sample_le

rule MF_CASCI:
    input: "geometry.xyz"
    output: hf="hf.chk",casci="casci.chk"
    run:
        with open(input[0],'r') as f:
            txt = f.read()
        sample_le.hf_casci(txt,output.hf, output.casci)


rule LINEAR: 
    input: hf="hf.chk",casci="casci.chk"
    output: 'linear.chk'
    run: 
        mc_calc = sample_le.gen_wf(input.hf,input.casci, {0:1.0})
        sample_le.linear(mc_calc, output[0])

rule EXCITED:
    input: hf="hf.chk",casci="casci.chk",anchor='linear.chk'
    output: "excited.chk"
    run:
        mc_ex = sample_le.gen_wf(input.hf,input.casci,{1:1.0})
        mc_ref = sample_le.restart_wf(input.hf, input.casci, input.anchor)
        sample_le.optimize_excited([mc_ref],mc_ex,Starget=[0.0],forcing=[2.0],hdf_file=output[0])

    