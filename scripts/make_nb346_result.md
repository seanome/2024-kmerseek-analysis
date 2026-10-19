**Result: KILL**, by the rule written before the search, which decides on the BCL-2/CED-9
pair in both directions. Searching whole proteomes, no HP score finds BCL-2 from CED-9 or
CED-9 from BCL-2, while phmmer (20 letters) ranks BCL-2 4th of 19,732 human proteins, behind
three other BCL-2 family members, and CED-9 1st of 19,792 *C. elegans* proteins. Under the
linked key BCL-2 ranks 9,952, and 48% of dipeptide-shuffled decoys score as high; CED-9
ranks 4,493, with a decoy rate of 21%. The linked key ranks the partner a little better
than B3 in both directions (9,952 against 10,798; 4,493 against 5,244), so prediction P2
failed and the KILL rests on the decoy-rate clause alone. BCL-2's best key over the two
whole sequences lies wholly off the structure alignment, so even its score is a chance
match. Real human proteins share 1.55 times the linked keys that notebook 294's
word-frequency count predicts for their search area. The best partial results: with CED-9's
crystallised domain as the query the randstrobe ranks BCL-2 1,058 (decoy rate 0.05), and
the best family member, MCL1, ranks 1,774 for full-length CED-9. Against Swiss-Prot (572,970
proteins), none of the four Pfam pairs keeps a decoy rate at or below 0.001 under the linked
key (prediction P5 held); the closest, PF18196, ranks 1,421 with a decoy rate of 0.003, and B3
does better on it (rank 986, 0.002). kmerseek's own chained search waits on a Sherlock job
whose first run ran out of memory; it cannot change the verdict.
