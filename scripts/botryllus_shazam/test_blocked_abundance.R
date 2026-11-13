# Software fixture only; never an experimental result.
args <- commandArgs(TRUE)
folder <- args[1]; code <- args[2]
dir.create(folder, recursive=TRUE, showWarnings=FALSE)
prefix <- if(length(args)>=3) args[3] else file.path(Sys.getenv("GROUP_HOME"),"opt/botryllus-dekupl-1.3.3-20261024")
raw_fixture <- readLines(file.path(dirname(folder),'native-fixture-raw.txt'),n=1)
nf_path <- file.path(folder,'native-normalization-check.tsv')
status_nf <- system2(file.path(prefix,'share/dekupl/bin/computeNF'),raw_fixture,stdout=nf_path)
if(status_nf!=0) stop('Native normalization program failed')
native_nf <- read.delim(nf_path)
if(!identical(names(native_nf),c('sample','normalization_factor')) || nrow(native_nf)!=4 || any(abs(native_nf$normalization_factor-1)>1e-6)) stop('Native normalization fixture differs')
set.seed(1729)
units <- expand.grid(condition=c('Control','Treatment'), genotype=c('G1','G2','G3'), stringsAsFactors=FALSE)
units$unit <- paste0('U',seq_len(nrow(units)))
write.table(units[,c('unit','genotype','condition')],file.path(folder,'units.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
base <- runif(2000,200,1000)
mu <- outer(base, rep(1,6))
for(j in 1:6) {
  mu[,j] <- mu[,j] * c(G1=0.7,G2=1,G3=1.4)[units$genotype[j]]
  if(units$condition[j]=='Treatment') mu[1:40,j] <- mu[1:40,j]*4
}
counts <- matrix(rnbinom(length(mu),mu=as.vector(mu),size=30),ncol=6)
colnames(counts) <- units$unit
write.table(data.frame(tag=paste0('fixture',1:2000),counts),gzfile(file.path(folder,'part-000.tsv.gz')),sep='\t',quote=FALSE,row.names=FALSE)
write.table(data.frame(sample=units$unit,normalization_factor=1),file.path(folder,'normalization.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
status <- system2(file.path(R.home('bin'),'Rscript'),c(file.path(code,'blocked_abundance.R'),folder,'software_fixture','genes'),stdout=file.path(folder,'test.log'),stderr=file.path(folder,'test.err'))
if(status!=0) stop('Blocked model fixture failed')
result <- read.delim(gzfile(file.path(folder,'all-features.tsv.gz')),check.names=FALSE)
if(nrow(result)!=2000 || anyDuplicated(result$tag)) stop('Fixture output coverage differs')
spikes <- result[match(paste0('fixture',1:40),result$tag),]
if(median(spikes$log2FoldChange,na.rm=TRUE)<1.5 || sum(spikes$padj<0.05,na.rm=TRUE)<20) stop('Blocked model failed obvious positive-control effects')
if(max(abs(p.adjust(result$pvalue,'BH',n=2000)-result$padj),na.rm=TRUE)>1e-12) stop('Global BH check differs')
writeLines('PASS: 2000 fixture genes, three genotype blocks, known treatment effects, independent global BH check',file.path(folder,'validated.txt'))
