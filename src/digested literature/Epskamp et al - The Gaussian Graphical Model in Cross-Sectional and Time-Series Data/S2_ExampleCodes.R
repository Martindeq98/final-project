# Load packages: 
library("mlVAR")
library("graphicalVAR")
library("qgraph")

# Read the data:
url <- "http://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0060188.s004&type=supplementary"
Data <- subset(read.table(url, header = TRUE, sep = ","), st_period == 0)[,-c(4,5,12)]

# Rename with English names:
names(Data) <- c("id","day","beep","cheerful","pleasant","worry","fearful","sad","relaxed")

# Input:
vars <- c("cheerful","pleasant","worry","fearful","sad","relaxed")
idvar <- "id"
# We do not need beepvar and dayvar arguments because the data is neatly pre-processed. Every beep is included even if responses are missing, and a row of NA responses is included between each day.

# Estimator 1: Mplus 8:
res_Mplus <- mlVAR(Data, vars, idvar, lags = 1,
                   temporal = "correlated", contemporaneous = "fixed",
                   estimator = "Mplus", nCores = 3)

pdf("Figure6_Mplus.pdf",width=9,height=3)
layout(t(1:3))
plot(res_Mplus, "temporal", layout = "circle", nonsig = "hide", theme = "colorblind", title = "Temporal",
     labels = vars, vsize = 20, rule = "and", asize = 10, mar = c(5,5,5,5))
box("figure")
plot(res_Mplus, "contemporaneous", layout = "circle", nonsig = "hide", theme = "colorblind", title = "Contemporaneous",
     labels = vars, vsize = 20, rule = "and", asize = 10, mar = c(5,5,5,5))
box("figure")
plot(res_Mplus, "between", layout = "circle", nonsig = "hide",  theme = "colorblind", title = "Between-subjects",
     labels = vars, vsize = 20, rule = "and", asize = 10, mar = c(5,5,5,5))
box("figure")
dev.off()

# Estimator 2: two-step mlvar:
res_mlVAR <- mlVAR(Data, vars, idvar, lags = 1,
             temporal = "correlated", contemporaneous = "correlated")

pdf("Figure6_mlVAR.pdf",width=9,height=3)
layout(t(1:3))
plot(res_mlVAR, "temporal", layout = "circle", nonsig = "hide", theme = "colorblind", title = "Temporal",
     labels = vars, vsize = 20, rule = "and", asize = 10, mar = c(5,5,5,5))
box("figure")
plot(res_mlVAR, "contemporaneous", layout = "circle", nonsig = "hide", theme = "colorblind", title = "Contemporaneous",
     labels = vars, vsize = 20, rule = "and", asize = 10, mar = c(5,5,5,5))
box("figure")
plot(res_mlVAR, "between", layout = "circle", nonsig = "hide",  theme = "colorblind", title = "Between-subjects",
     labels = vars, vsize = 20, rule = "and", asize = 10, mar = c(5,5,5,5))
box("figure")
dev.off()

# Via graphicalVAR:
library("graphicalVAR")
res_GVAR <- mlGraphicalVAR(Data, vars, idvar = idvar,gamma = 0.25, subjectNetworks = FALSE)

pdf("Figure6_GVAR.pdf",width=9,height=3)
layout(t(1:3))
qgraph(res_GVAR$fixedPDC,  layout = "circle", nonsig = "hide", theme = "colorblind", title = "Temporal",
       labels = vars, vsize = 20,  asize = 10, mar = c(5,5,5,5))
box("figure")
qgraph(res_GVAR$fixedPCC,  layout = "circle", nonsig = "hide", rule = "and", theme = "colorblind", title = "Contemporaneous",
       labels = vars, vsize = 20, asize = 10, mar = c(5,5,5,5))
box("figure")
qgraph(res_GVAR$betweenNet, layout = "circle", nonsig = "hide", rule = "and", theme = "colorblind", title = "Between-subjects",
       labels = vars, vsize = 20,  asize = 10, mar = c(5,5,5,5))
box("figure")
dev.off()


