**Result: KILL**, by the rule written before the search. Searching whole proteomes, no
HP score finds BCL-2 from CED-9 or CED-9 from BCL-2. Under the linked key BCL-2 ranks
9,952 of 19,732 human proteins, and 48% of dipeptide-shuffled decoys score as high. CED-9
ranks 4,493 of 19,792 *C. elegans* proteins, with a decoy rate of 21%. The linked key ranks
the true partner a little better than B3 in both directions (9,952 against 10,798; 4,493
against 5,244), so prediction P2 failed. But both decoy rates are far above the 0.01 KILL
line. BCL-2's best key over the two whole sequences lies wholly off the structure
alignment, so even its score is a chance match. Each human protein shares a median of
41,720 linked keys with CED-9 over all diagonals; notebook 294 counted 3,896 for two
131-residue proteins on one diagonal. Pairs 3 to 6 against Swiss-Prot (prediction P5) and
kmerseek's own chained search wait on two Sherlock jobs.
