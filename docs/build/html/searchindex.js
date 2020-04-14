Search.setIndex({docnames:["index","src","src.estimation","src.estimation.base","src.estimation.cca","src.estimation.cosine","src.estimation.euclidean","src.estimation.prob","src.experiments","src.experiments.visualize_nn_progress","src.experiments.word_rec","src.experiments.word_spotting","src.io","src.io.dataloader","src.nn","src.nn.phocnet","src.nn.pp","src.nn.stn","src.parser","src.parser.args_parser","src.parser.to_data","src.training","src.training.cca_cross_validation","src.training.cca_trainer","src.training.phocnet_trainer","src.util","src.util.alphabet_chars","src.util.augmentation_util","src.util.eval_util","src.util.phoc_util","src.util.representation_util","src.util.sanity_util"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["index.rst","src.rst","src.estimation.rst","src.estimation.base.rst","src.estimation.cca.rst","src.estimation.cosine.rst","src.estimation.euclidean.rst","src.estimation.prob.rst","src.experiments.rst","src.experiments.visualize_nn_progress.rst","src.experiments.word_rec.rst","src.experiments.word_spotting.rst","src.io.rst","src.io.dataloader.rst","src.nn.rst","src.nn.phocnet.rst","src.nn.pp.rst","src.nn.stn.rst","src.parser.rst","src.parser.args_parser.rst","src.parser.to_data.rst","src.training.rst","src.training.cca_cross_validation.rst","src.training.cca_trainer.rst","src.training.phocnet_trainer.rst","src.util.rst","src.util.alphabet_chars.rst","src.util.augmentation_util.rst","src.util.eval_util.rst","src.util.phoc_util.rst","src.util.representation_util.rst","src.util.sanity_util.rst"],objects:{"":{src:[1,0,0,"-"]},"src.estimation":{base:[3,0,0,"-"],cca:[4,0,0,"-"],cosine:[5,0,0,"-"],euclidean:[6,0,0,"-"],prob:[7,0,0,"-"]},"src.estimation.base":{DistEstimator:[3,1,1,""],Estimator:[3,1,1,""],Metrices:[3,1,1,""]},"src.estimation.base.DistEstimator":{dists:[3,2,1,""],estimate:[3,2,1,""],estimate_set:[3,2,1,""],process_of_measure:[3,2,1,""],words:[3,2,1,""]},"src.estimation.base.Estimator":{est_word:[3,2,1,""],estimate:[3,2,1,""],estimate_set:[3,2,1,""],n_neighbour:[3,2,1,""],process_of_measure:[3,2,1,""],retrieval_list:[3,2,1,""],save:[3,2,1,""],words:[3,2,1,""]},"src.estimation.base.Metrices":{COSINE:[3,3,1,""],EUCLIDEAN:[3,3,1,""],MAHALANOBIS:[3,3,1,""]},"src.estimation.cca":{RCCAEstimator:[4,1,1,""]},"src.estimation.cca.RCCAEstimator":{estimate_set:[4,2,1,""],fit:[4,2,1,""],nn_search_idcs:[4,2,1,""],norm:[4,2,1,""],process_of_measure:[4,2,1,""],save:[4,2,1,""],transform:[4,2,1,""],words:[4,2,1,""]},"src.estimation.cosine":{CosineEstimator:[5,1,1,""]},"src.estimation.euclidean":{EuclideanEstimator:[6,1,1,""]},"src.estimation.prob":{ProbEstimator:[7,1,1,""]},"src.estimation.prob.ProbEstimator":{estimate:[7,2,1,""],prm_scores:[7,2,1,""],process_of_measure:[7,2,1,""],words:[7,2,1,""]},"src.experiments":{visualize_nn_progress:[9,0,0,"-"],word_rec:[10,0,0,"-"],word_spotting:[11,0,0,"-"]},"src.experiments.visualize_nn_progress":{evaluate_dir:[9,4,1,""],plot_series:[9,4,1,""]},"src.experiments.word_rec":{run_word_rec:[10,4,1,""],save:[10,4,1,""]},"src.experiments.word_spotting":{attr_vecs:[11,4,1,""],load_net:[11,4,1,""],run_wordspotting:[11,4,1,""],write_map:[11,4,1,""]},"src.io":{dataloader:[13,0,0,"-"]},"src.io.dataloader":{DSetPhoc:[13,1,1,""],DSetQuant:[13,1,1,""],GWDataSet:[13,1,1,""],HWSynthDataSet:[13,1,1,""],IAMDataset:[13,1,1,""],RimesDataSet:[13,1,1,""],SubSet:[13,1,1,""],quant_to_rep:[13,4,1,""],rep_to_quant:[13,4,1,""]},"src.io.dataloader.DSetPhoc":{apply_alphabet:[13,2,1,""],augment:[13,2,1,""],bbox:[13,2,1,""],bbox_img:[13,2,1,""],bbox_list:[13,2,1,""],cls_to_idcs:[13,2,1,""],display:[13,2,1,""],exclude_words:[13,2,1,""],form:[13,2,1,""],form_path_list:[13,2,1,""],id:[13,2,1,""],ids:[13,2,1,""],img:[13,2,1,""],inv_img:[13,2,1,""],needs_lower:[13,2,1,""],norm_img:[13,2,1,""],phoc:[13,2,1,""],phoc_list:[13,2,1,""],sub_set:[13,2,1,""],transcript:[13,2,1,""],transcript_idcs:[13,2,1,""],word_list:[13,2,1,""],words:[13,2,1,""]},"src.io.dataloader.DSetQuant":{EQUAL:[13,3,1,""],RANDOM:[13,3,1,""],RESPECTIVE:[13,3,1,""]},"src.io.dataloader.GWDataSet":{fold:[13,2,1,""]},"src.io.dataloader.HWSynthDataSet":{needs_lower:[13,2,1,""],train_test_official:[13,2,1,""]},"src.io.dataloader.IAMDataset":{id_to_form_f_name:[13,2,1,""],id_to_path:[13,2,1,""],needs_lower:[13,2,1,""],test_set_official:[13,2,1,""],train_set_official:[13,2,1,""],train_test_official:[13,2,1,""],word_id_to_line_id:[13,2,1,""]},"src.io.dataloader.RimesDataSet":{needs_lower:[13,2,1,""],reload_rimes:[13,2,1,""],to_ascii:[13,2,1,""],train_test_official:[13,2,1,""]},"src.nn":{phocnet:[15,0,0,"-"],pp:[16,0,0,"-"],stn:[17,0,0,"-"]},"src.nn.phocnet":{PHOCNet:[15,1,1,""],STNPHOCNet:[15,1,1,""]},"src.nn.phocnet.PHOCNet":{convolute:[15,2,1,""],display_forward:[15,2,1,""],forward:[15,2,1,""],init_weights:[15,2,1,""],linear_act:[15,2,1,""],linear_dropout:[15,2,1,""],linear_sigmoid:[15,2,1,""],neural_codes:[15,2,1,""],pool:[15,2,1,""],setup:[15,2,1,""]},"src.nn.phocnet.STNPHOCNet":{neural_codes:[15,2,1,""]},"src.nn.pp":{GPP:[16,1,1,""],PPTypePooling:[16,1,1,""],PPTypes:[16,1,1,""]},"src.nn.pp.GPP":{forward:[16,2,1,""],gpp_type:[16,2,1,""],pool_type:[16,2,1,""]},"src.nn.pp.PPTypePooling":{AVG_POOL:[16,3,1,""],MAX_POOL:[16,3,1,""]},"src.nn.pp.PPTypes":{T_SPP:[16,3,1,""],T_TPP:[16,3,1,""]},"src.nn.stn":{STN:[17,1,1,""]},"src.nn.stn.STN":{T_theta:[17,2,1,""],f_loc:[17,2,1,""],forward:[17,2,1,""],pool:[17,2,1,""],sampler:[17,2,1,""],setup:[17,2,1,""]},"src.parser":{args_parser:[19,0,0,"-"],to_data:[20,0,0,"-"]},"src.parser.args_parser":{parser_inference:[19,4,1,""],parser_training:[19,4,1,""]},"src.parser.to_data":{get_PHOCNet:[20,4,1,""],get_dsets:[20,4,1,""],get_estimator:[20,4,1,""]},"src.training":{cca_cross_validation:[22,0,0,"-"],cca_trainer:[23,0,0,"-"],phocnet_trainer:[24,0,0,"-"]},"src.training.cca_cross_validation":{cca_run:[22,4,1,""],cross_val:[22,4,1,""],gather_NC_TRANS_pairs:[22,4,1,""],w_err_CCA:[22,4,1,""]},"src.training.cca_trainer":{equal_split:[23,4,1,""],gather_NC_PHOC_pairs:[23,4,1,""],parser:[23,4,1,""],run_cca_training:[23,4,1,""],train_cca:[23,4,1,""]},"src.training.phocnet_trainer":{CosineLoss:[24,1,1,""],Trainer:[24,1,1,""],adam_optimizer:[24,4,1,""],new_logger:[24,4,1,""],sgd_optimizer:[24,4,1,""]},"src.training.phocnet_trainer.CosineLoss":{forward:[24,2,1,""]},"src.training.phocnet_trainer.Trainer":{device:[24,2,1,""],save:[24,2,1,""],set_up:[24,2,1,""],train_on:[24,2,1,""],train_on_batch:[24,2,1,""]},"src.util":{alphabet_chars:[26,0,0,"-"],augmentation_util:[27,0,0,"-"],eval_util:[28,0,0,"-"],phoc_util:[29,0,0,"-"],representation_util:[30,0,0,"-"],sanity_util:[31,0,0,"-"]},"src.util.alphabet_chars":{dset_chars:[26,4,1,""],parser:[26,4,1,""]},"src.util.augmentation_util":{homography_augm:[27,4,1,""],scale:[27,4,1,""],visualiz_homography_augm:[27,4,1,""]},"src.util.eval_util":{ap:[28,4,1,""],map:[28,4,1,""],overlap:[28,4,1,""],relevance:[28,4,1,""],ret_list_idcs:[28,4,1,""]},"src.util.phoc_util":{Alphabet:[29,1,1,""],alphabet_chars:[29,4,1,""],alphabet_to_rep:[29,4,1,""],char_err:[29,4,1,""],hoc:[29,4,1,""],is_occ:[29,4,1,""],len_phoc:[29,4,1,""],occ:[29,4,1,""],occ_abs:[29,4,1,""],occ_intersect:[29,4,1,""],phoc:[29,4,1,""],phoc_levels:[29,4,1,""],rep_to_alphabet:[29,4,1,""],word_err:[29,4,1,""]},"src.util.phoc_util.Alphabet":{ASCII_DIGITS:[29,3,1,""],ASCII_LOWER:[29,3,1,""],ASCII_PUNCTUATION:[29,3,1,""],ASCII_UPPER:[29,3,1,""],PERFECT_GW:[29,3,1,""],PERFECT_IAM:[29,3,1,""],PERFECT_RIMES:[29,3,1,""]},"src.util.representation_util":{js_stylize:[30,4,1,""],jsons_to_table:[30,4,1,""],parser:[30,4,1,""]},"src.util.sanity_util":{np_arr:[31,4,1,""],safe_dir_path:[31,4,1,""],unique_file_name:[31,4,1,""]},src:{estimation:[2,0,0,"-"],experiments:[8,0,0,"-"],io:[12,0,0,"-"],nn:[14,0,0,"-"],parser:[18,0,0,"-"],training:[21,0,0,"-"],util:[25,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"1e5":[13,21,24],"abstract":3,"case":[20,22],"char":29,"class":[1,3,4,5,6,7,13,14,15,16,17,23,24,29],"default":[11,19,24],"enum":[3,13,16,29],"final":[1,2,24],"float":29,"function":[3,19],"int":29,"new":[13,26,27],"return":[3,4,7,9,10,11,13,15,16,17,19,20,22,23,24,26,27,28,29,30],"static":[3,13,15],"true":[4,10,13,19,22,23,24],"while":[2,27],DBs:13,For:[9,10,11,17],NOT:[3,4],That:2,The:[1,2,3,4,7,8,10,13,14,15,17,24],Their:[3,9],There:10,These:31,Use:3,__getitem__:13,_word:3,absolut:29,accor:3,accord:[3,9],activ:15,adam:[19,24],adam_optim:24,adapt:[12,13,26],add:31,addition:4,affect:3,affin:27,after:4,alik:[3,4,7],all:[1,3,4,8,9,12,13,15,21,24,25,26,29,30],allow:13,allwai:13,almazan:[13,19,21,22,24],almost:3,along:19,alph_char:29,alphabet:[3,4,5,6,7,13,19,20,21,22,23,24,25,26,29],alphabet_char:[1,25,29],alphabet_to_rep:29,alreadi:[1,31],also:[13,21,24],altern:[1,13],alwai:2,analysi:13,ani:[3,4,7,13,15,25],annot:[13,20,26],appli:[2,4,17],apply_alphabet:13,approach:[2,4,7,13,14,17],architectur:15,area:28,arg:[1,18,19,30],args_pars:[1,9,10,11,18,21,24],argument:[13,19,20,24,30],arr_bbox_est:28,arr_bbox_gt:28,arr_form_est:28,arr_form_gt:28,arrai:[3,4,29],arror:9,ascii:13,ascii_digit:[3,4,5,6,7,13,22,23,24,29],ascii_low:[3,4,5,6,7,13,22,23,24,29],ascii_punctu:[3,4,5,6,7,13,24,29],ascii_upp:[22,23,29],assumpt:28,aswel:[20,24],atribut:7,attr_vec:[3,11,28],attribut:[1,2,3,4,7,11,15,29],auf:13,augemntation_util:25,augment:[13,19,23,24,25,27],augment_img:13,augmentation_util:[1,13,25],avail:1,averag:28,avg_pool:16,axi:19,bachelor:1,backward:24,base:[1,2,4,5,6,7,9,10,11,13,15,16,17,24,28,29],batch:[4,9,19,22,24],bbox1:28,bbox2:28,bbox:[13,28],bbox_img:[13,28],bbox_list:13,bceloss:24,been:[22,24,27,31],befor:[2,15],behaviour:[3,4],besst:22,better:[4,13,25,30],between:[15,19],bin:16,bin_relev:28,binari:28,boolena:29,both:27,bound:[13,28],box:[13,28],brief:1,build:2,calcul:[3,7,11,15,22,24,28,29],call:3,can:[1,8,12,13,14,15,17,26,31],caract:29,carri:4,cca:[1,2,15,22,23],cca_cross_valid:[1,21],cca_run:22,cca_train:[1,21],cdist:3,cer:[10,25,29],certaint:13,char_err:29,charact:[9,10,25,26,29],choos:19,clalcul:29,classic:[7,19],clean:[3,4],close:11,cls_to_idc:13,cnn:7,code:[1,2,3,4,10,15,19,22,23],collumn:13,come:13,compar:[3,4,7],comparison:[3,4,7],comput:[11,27],concaten:16,config:24,configur:[4,22,24],consist:1,construct:[1,2,18],contain:[10,13,24,25,29,31],content:0,conv2d:15,conv:15,convolut:15,corpora:13,correctli:3,correspond:[7,13,22],cosin:[1,2,3,4,8,9,10,11,19,24,28],cosineestim:[5,9],cosineloss:24,cpu:11,creat:[1,11,15,27,29,30],creation:25,cross:[13,19,22],cross_val:22,csv:19,csvs_path:13,cuda:[8,9,10,11,13,19,21,22,24],current:[1,7],custom:[15,23],cut:13,d_set:24,data:[1,3,4,10,12,13,15,19,24,26,27,30],databas:13,datafram:13,dataload:[1,10,11,12,22,23,24,26],dataset:[1,4,11,13,19,20,22,23,24,26],deal:[1,8,13,25],debug:[10,13],defin:[2,7,27],deivc:11,demand:[3,4],depend:[1,13],desir:[3,28],determin:29,determinist:29,devic:[9,10,11,19,22,23,24],dict:19,dictionari:[10,13,24],differ:[1,13,14,16,22],dim:[22,23],dim_phoc:3,dimens:[22,23],dir:[3,4,9,19,31],dir_json:30,dir_out:[8,9,10,11,13,19,21,22,24,30],dir_path:31,direct:[2,3,4],directli:13,directori:[3,4,8,9,10,11,13,19,24,30,31],discuss:17,disk:31,dispar:4,displai:[13,15],display_forward:15,dist:3,distanc:[1,3,4,5,6,7,29],distestim:[3,5,6],distribut:23,document:13,doe:15,driven:2,dropout:15,dset:[10,11,13,22,23,26],dset_annot:[8,9,10,11],dset_char:26,dset_csv:[19,20],dset_img:20,dset_nam:[8,9,10,11,19,20],dset_src:19,dset_test:[9,22],dset_train:22,dsetphoc:[10,11,13,22,23,24],dsetquant:[13,24],duck:3,due:4,dure:[8,19,24],dynam:29,each:24,eas:[18,20],easi:[3,13],element:28,embed:[1,24],encod:[1,13,15,29],ensur:3,entir:4,enumer:[13,31],epoch:24,equal:[13,23,24,29],equal_split:23,equival:10,err:9,erro:10,error:[9,10,22,29],essenc:1,est_nam:20,est_phoc:7,est_word:3,establish:20,estim:[0,1,8,9,10,11,15,19,20,21,22,23,24,29],estimate_set:[3,4,7],etsablish:20,euclidean:[1,2,3],euclideanestim:6,eugen:7,eval_util:[1,25],evalu:[3,8,9,10,11,25,28],evaluate_dir:9,evei:1,evenli:23,everi:1,exampl:[8,9,10,11,21,22,24],exclud:13,exclude_word:13,exist:[3,4,15,16,31],experi:[0,1],explicitli:[3,4,7],extra:19,extract:[1,10,13,15,19,22,23,25,30],f_loc:17,f_name:11,fact:15,fals:[3,10,13,23,24],feasibl:25,featur:[1,17],feature_map:17,file:[3,4,8,9,10,11,13,24,25,30,31],file_nam:30,filenam:31,filter:16,first:29,fit:[4,13],fix:22,fki:13,flag:19,float32:24,focus:2,fold:[13,19,20,22],follow:[1,13,17,19,30],form:13,form_path:13,form_path_list:13,format:30,forward:[10,15,16,17,24],fowardpass:15,framework:[12,14,26],french:13,frequenc:19,from:[1,10,13,15,16,19,22,23,27,28,29,30],front:15,gather:[11,22,23],gather_nc_phoc_pair:23,gather_nc_trans_pair:22,gener:[2,13,16,17,24,27,29],georg:[1,12,13,20],get_dset:20,get_estim:20,get_phocnet:20,given:[1,23,24,28,29],global:13,good:17,good_segment:13,gpp:16,gpp_type:16,gpu:[9,10,11,22,23],gpu_idx:[8,9,10,11,13,19,21,22,24],greatli:3,grid:17,groundtruth:13,gtp:[13,21,22,24],guarante:17,gw_databas:[21,22,24],gwdataset:[13,26],gwdb:[13,21],handl:10,happen:11,happend:13,has:[24,31],have:[4,7,9,10,11,15,17,22,24,27],height:27,henc:[1,3,4,7,17,24],here:1,highest:7,his:15,hit:28,hoc:[22,29],homographi:27,homography_augm:[13,27],horizont:16,how:[3,4,7],html:30,http:13,hw_synth:13,hws:13,hwsynth:13,hwsynthdataset:13,hybrid:23,hyper:[3,22,23],hyperparamet:22,iam:[1,11,12,13,19],iamdataset:13,iamdb:13,id_to_form_f_nam:13,id_to_path:13,idc:[13,28],ident:3,ids:13,idx:13,iff:31,iiit:13,imag:[1,10,11,13,15,17,19,20,21,22,23,24,25,26,27],images_90k_norm:13,img:[8,9,10,11,13,27],img_id:13,imgs_path:13,immut:7,implement:[3,4,14,15,16,17,19,24,27],implicitli:7,implment:16,indesx:13,index:[3,13,19,29],indic:[3,4,13,19,24,28,29],individu:[4,9],inf:13,infer:[1,8,9,18,19,22,23],infield:13,infom:22,inform:[15,17,22,23,30],infrom:22,init_weight:15,initi:[9,14,15,17,19,24],input:[15,16,17],input_channel:[15,17],input_var:24,input_x:16,inspir:[15,16],instanc:[3,4,7,9,10,13],interpret:[2,18,20],intersect:29,interv:[19,29],introduc:3,intv:29,intv_0:29,intv_1:29,intv_char:29,intv_reg:29,inv_img:13,invert:13,is_occ:29,item:13,iter:[13,19,24],its:[3,4,24,29],jaderberg:17,js_styliz:30,json:[4,8,9,10,11,24,25,30],json_dict:10,jsons_to_t:30,k_fold:[13,19,20,21,22,24],keep:9,kept:27,kind:19,larg:3,larger:15,largewriterindependenttextlinerecognitiontask:13,layer:[14,15,16,17,19],lazy_load:13,ldp:[13,19,21,22,24],lead:13,learn:19,leav:15,len_phoc:29,length:29,level:[10,16,19,20,22,23,29],lexicon:[1,2,3,4,7,20,22,23],like:4,line:13,linear:15,linear_act:15,linear_dropout:15,linear_sigmoid:15,link:13,list:[3,4,9,13,23,28,29],load:11,load_net:11,localis:17,locat:9,log:[4,8,24],logger:[9,22,23,24],look:[9,10,11,17],loop:24,loss:[19,24],lower_cas:13,mahalanobi:3,mai:2,main:10,make:[13,31],mani:25,manual:[3,4,19],map:[11,13,17,28,29],matrix:3,max:17,max_it:[13,19,21,24],max_pool:[15,16],max_sample_s:23,maximum:[19,23],mean:[4,9,10,22,28],measur:[1,2,3,4,7],mention:26,meta:24,metadata:19,method:[1,3,4,7,8,10,13,15,18,19,20,22,23,25,26,27,29,31],metric:[2,3,4,5,6,28],mind:11,misclassifi:22,miss:4,mixed_precis:24,model:[1,2,7,8,9,10,11,14,15,19,24],model_nam:[13,19,21,22,24],modul:0,momentum:24,more:[15,17],mspring:13,much:[3,4,7,15],mutat:[3,4],my_model:19,my_phocnet:[21,24],my_rcca:[21,22],n_code:22,n_code_lvl:[22,23],n_codes_lvl:[9,10,11,19],n_dim:4,n_f_map:16,n_fold:22,n_iter:24,n_neighbour:3,n_occ:28,n_out:15,n_sampl:3,name:[3,4,9,10,11,13,19,20,24,28,30,31],nativ:24,ndarrai:[3,7],nearest:[1,2,3,4],need:[1,13,23],needs_low:13,neighbour:[1,2,3,4],net:[8,10,11,19,22,23,24],net_log_dir:24,net_path:[8,11,19],network:[15,17],neural:[1,2,3,4,8,10,15,19,22,23],neural_cod:[15,22,23],new_logg:24,nn_my_phocnet:21,nn_path:11,nn_search_idc:4,non_aug_img:13,none:[3,10,11,13,15,19,23,24,27],norm:4,norm_img:13,normal:[4,10,13,15,19,20],note:[3,4,13],notmal:15,np_arr:31,number:[19,20,22,24,29],numpi:[3,7],object:[3,10,13,23,24],obtain:13,occ:[28,29],occ_ab:29,occ_intersect:29,occup:29,occupi:[28,29],occur:13,offici:13,offset:15,one:[15,27],onli:[3,13,16,27],optim:[19,24],optimis:3,option:[8,9,10,11,13,15,19,21,24],order:[3,9,13,15,22,23,28,29],origin:[13,27],other:15,otherwis:13,our:2,out:4,outdat:[3,4],output:[8,9,10,11,15,19,24,26,30],output_dir:[21,24],over:9,overlap:[28,29],overrid:15,overview:[1,30],overwritten:[3,4],own:24,packag:0,page:[13,28,30],paper:[15,17,24],paraamet:22,paramet:[3,4,7,9,10,11,13,15,16,17,20,22,23,24,26,27,28,29,30],pars:[13,29],parser:[0,1,9,10,11,21,23,24,26,30],parser_infer:[9,10,11,19],parser_train:[19,21,24],pass:[8,10,13,15,17,20],path:[8,9,10,11,13,19,20,21,22,24,25,26,28],pct:29,per:29,percentag:28,perfect_gw:29,perfect_iam:29,perfect_rim:29,perfom:15,perform:[1,3,4,10,11,15,24,27],pfx:24,phoc:[1,2,3,4,7,11,13,15,19,20,22,23,25,29],phoc_level:[3,5,6,7,13,29],phoc_list:[13,28],phoc_lvl:[4,19,20,24],phoc_util:[1,13,19,25],phocnet:[1,9,10,11,14,16,17,19,20,21,22,23,24],phocnet_statedict:22,phocnet_train:[1,13,21],phocnet_typ:19,pickl:[19,20],pikl:[3,4],pip:1,pipelin:1,place:[22,23],pleas:[3,4],plot:9,plot_seri:9,png:13,point:27,pool:[14,15,16,17],pool_typ:[15,16],pooling_level:15,pos:15,posit:[15,19,30],possibl:3,pp_type:15,pptype:[15,16],pptypepool:[15,16],pre:[3,4,15,16],precis:28,predict:[2,15],pretrain:19,pretrained_phocnet:13,print:[10,26],prm:[1,2,7],prm_score:7,prob:[1,2],probabilist:[2,7],probabl:[2,7],probestim:7,procedur:16,process:[1,2,3,4,7,9,19,24],process_of_measur:[3,4,7],program:29,progress:[8,9],prohibit:[3,4],project:1,propag:[15,24],properti:[3,4,7,13,16,19,24,29],proport:10,propos:[2,7,15,24,27],provid:[1,2,4,5,6,8,10,11,12,13,14,17,18,19,20,21,22,23,24,26,27,28,29,30],pth:21,punctuat:19,pyramid:[14,16],pyrcca:[1,4],python3:[8,9,10,11,13,21,22,24,26],python:7,pytorch:[13,24],quant:13,quant_aug:24,quant_to_rep:13,quantif:13,queri:[3,4,13,21,22,24],random:13,random_limit:27,randomli:27,rang:19,rate:[10,19],ratio:27,rcca:[21,22,23],rccaestim:[4,22],real:13,recognit:[1,2,8,10,11],rectangular:28,refer:[3,4],reg:[4,22,23],regard:[1,3,13,21,24],region:29,regular:[2,4,15,17,22,23],regularis:[4,22],rel:29,relat:11,relev:[12,13,24,28,29],reload:13,reload_rim:13,relu:15,rep:13,rep_to_alphabet:[19,29],rep_to_qu:13,repositori:13,repres:[3,4,7,29],represent:[1,3,13,25],representation_util:[1,25],request:29,resid:1,resiz:27,respect:[1,3,4,7,9,10,13,22,23,24,29],restpect:11,result:[8,15,17],ret_list_idc:28,retriev:[2,3,7,28],retrieval_list:3,retriv:28,retsina:24,rime:[1,11,12,13],rimesdataset:13,run:[8,9,19,22,23,24],run_cca_train:23,run_word_rec:10,run_wordspot:11,runtim:3,rusakov:7,s_aug:24,s_batch:[9,13,19,21,22,24],safe:[19,30],safe_dir_path:31,said:1,same:[4,13],sampl:[4,10,17],sampler:17,sampling_grid:17,saniti:[25,31],sanity_util:[1,25],save:[3,4,10,19,24],save_interv:19,scale:[10,13,19,20,23,27],scale_h:19,scale_w:19,score:[1,2,7],script:[1,8,9,10,11,21,24,26,30],search:[1,2,3,4,5,6,13],sebastian:[15,27],second:29,see:[2,4,13,19,21,22,23,24,29],select:13,self:[3,13],semant:[19,29,30],sens:2,sep:13,sequenc:28,seriliz:10,set:[1,3,4,7,9,10,11,12,13,19,26,27],set_up:24,setup:[15,17],sgd:[19,24],sgd_optim:24,shall:[3,7,24],shape:3,share:1,should:[4,13],shuffl:[13,19,24],sigmoid:15,simmilarli:4,simpl:[15,17],simpli:8,sinc:23,singl:[3,8,23,29],size:[4,9,13,17,19,22,23],size_averag:24,slightli:7,slower:7,sole:3,some:[1,2,15,16],sometim:13,sort:[9,13,28],sourc:[3,4,5,6,7,9,10,11,13,15,16,17,19,20,22,23,24,26,27,28,29,30,31],space:3,spatial:[14,16,17],specif:30,specifi:[1,13,20,24,27],split:[13,20,23],spot:[1,2,11,25,28],spp:16,standard:24,state:[8,9,11,19,24],state_dict:[8,9,10,11],statedict:19,step:2,stn:[1,14,15,19,20],stnphocnet:[15,17],stop:19,stop_word:[13,19],store:[13,30],str:[13,29],string:[7,13,29],sub_set:13,subclass:[3,13],submodul:[0,1],subpackag:0,subset:[13,23],subspac:[1,2,4,15,22],substr:29,success:1,sudhold:15,sudholt:27,suffix:31,suggest:13,sum:24,sure:31,synth:[1,12,13],t_phocnet:20,t_quant:13,t_spp:16,t_theta:17,t_tpp:[15,16],tabl:[13,30],taken:31,target_var:24,tarnscript:13,task:[8,13],tatoal:10,tempor:16,tensor:15,test:[4,9,10,13,20,22,24],test_set_offici:13,than:4,thei:9,them:[9,30],themself:15,thereof:[1,14,18],thesi:[1,17],theta:17,thi:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],those:15,three:27,threshold:29,through:17,ths:13,titl:9,tmp:24,tmp_save_mod:24,to_ascii:13,to_data:[1,18],took:[15,16],torch:[12,13,14,15,16,17,24],tpp:16,track:9,train:[0,1,4,8,13,18,19,20],train_cca:23,train_data:3,train_on:24,train_on_batch:24,train_set_offici:13,train_test_offici:13,trainer:24,transcipt:13,transcript:[1,10,13,22,28],transcript_idc:13,transform:[1,2,4,14,17,27],tupl:9,type:[3,13,16,20,29],uint8:29,unambigu:3,unaug:13,underli:[3,4,7],unib:13,uniqu:[2,3,13,16,26],unique_file_nam:31,unit:4,unless:[3,4,7],unneed:[3,4],unwant:[3,4],use:[8,9,15,19,22],use_sigmoid:24,used:[1,3,4,7,9,10,12,14,15,17,19,20,23,24,26,29],useful:[13,25,26,31],uses:7,using:[1,2,3,13,15,17,29],usual:17,util:[0,1,2,13,19],valid:[13,19,22],validatin:22,valu:[3,4,7,13,16],vari:2,variabl:13,varianc:4,variat:1,vec:3,vecotr:3,vector:[1,3,4,7,11,15,16,24,29],version:16,vertic:16,via:[1,7,13],visual:[1,8,13,15],visualiz_homography_augm:27,visualize_nn_progress:[1,8],vol:13,w_err_cca:22,wai:29,warp:17,washingt:13,washington:[1,12,13,20],weight:15,well:[1,3,10,29],wer:[10,25,29],when:[3,4,13,26,31],where:[9,13],whether:[4,19,23,24,29],width:27,without:13,word:[1,2,3,4,5,6,7,8,9,10,11,13,15,19,20,22,23,25,28,29],word_err:29,word_id:13,word_id_to_line_id:13,word_img:13,word_list:13,word_rec:[1,8],word_spot:[1,8],word_str:13,work:4,would:13,wrap:3,write:[8,11,13,31],write_map:11,written:[9,10,24],www:13,x_in:[15,17],you:[3,15,31],your:3,zero:4,zip:13},titles:["src","src package","src.estimation package","src.estimation.base module","src.estimation.cca module","src.estimation.cosine module","src.estimation.euclidean module","src.estimation.prob module","src.experiments package","src.experiments.visualize_nn_progress module","src.experiments.word_rec module","src.experiments.word_spotting module","src.io package","src.io.dataloader module","src.nn package","src.nn.phocnet module","src.nn.pp module","src.nn.stn module","src.parser package","src.parser.args_parser module","src.parser.to_data module","src.training package","src.training.cca_cross_validation module","src.training.cca_trainer module","src.training.phocnet_trainer module","src.util package","src.util.alphabet_chars module","src.util.augmentation_util module","src.util.eval_util module","src.util.phoc_util module","src.util.representation_util module","src.util.sanity_util module"],titleterms:{alphabet_char:26,args_pars:19,augmentation_util:27,base:3,cca:4,cca_cross_valid:22,cca_train:23,content:[1,2,8,12,14,18,21,25],cosin:5,dataload:13,estim:[2,3,4,5,6,7],euclidean:6,eval_util:28,experi:[8,9,10,11],modul:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],packag:[1,2,8,12,14,18,21,25],parser:[18,19,20],phoc_util:29,phocnet:15,phocnet_train:24,prob:7,representation_util:30,sanity_util:31,src:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],stn:17,submodul:[2,8,12,14,18,21,25],subpackag:1,to_data:20,train:[21,22,23,24],util:[25,26,27,28,29,30,31],visualize_nn_progress:9,word_rec:10,word_spot:11}})