> Written by Claude Code (checker agent), 2026-10-18. A second agent, separate from the
> one that wrote the report, reopened every source and redid every sum. Its table is
> below as it returned it. The last section says what the report's author did with each
> correction.

# Second-agent check of `docs/edna_market/README.md`

## Summary

On 2026-10-18 the checker opened every linked source again and saved each one to a new,
empty scratch folder. Wiley blocked Jeunen 2023, so its full text was read from the
Liverpool John Moores repository copy. Every sum was redone in Python. Most numbers held:
the prices, the Teixeira 87%, Jeunen 574/1,450, Cerrillo-Espinosa 1,694/5,495, Jaquier
45/264, Norway's 995 sites, PondNet's 380 ponds, the Hull quote, the Farmers Weekly
quote and all the arithmetic.

Seven things failed in the first draft:
1. The Australian guide's first author is De Brauwer, not Suter.
2. Only three of the five studies were lower bounds, not four.
3. Saenz-Agudelo 2024 gives the genus share directly in its Table 3 (74.4% COI, 53.6%
   18S, 14.8% 16S of ASVs not named at genus). The draft used a weaker phylum-level
   range (8% to 38%).
4. Wilderlab's methods use Wilkinson's `insect` package to sort reads by sample barcode,
   not to name them.
5. Wilderlab's COI assay is in the basic panel, so the USD 68 comprehensive-minus-basic
   step pays for plant, microbe and other assays, not for COI. Every listed panel
   includes COI.
6. Bockrath is listed first on the FWS web page, but the guide has no overall first
   author; she is first author of chapter 6.
7. The Hull case study never names 12S.

The step 5 verdict held only under a loose reading of the rule: the draft showed the
pass could reach USD 50,000 if 736 samples a year (3.7%) bought it, not that it does.
USD 68 is a step between panel tiers, not an add-on. Wilderlab's own running total (over
50,000 samples by October 2024, founded 2019) is about 8,700 a year on average, below
the 20,000 used; even then 736 is only 8.5%.

## Table

| Claim (short) | Source | Reopened? | Report said | Source says / recomputed | Held or Failed |
|---|---|---|---|---|---|
| FWS best-practice guide, 2023 | fws.gov page | yes | 2023 | "Publication date Aug 21, 2023" (PDF file name says "Final Draft 4.1.22") | Held |
| FWS guide first author | fws.gov page + PDF | yes | "First author of the 2023 guide" | Page lists Bockrath first of 11. The PDF has authors per chapter: Bockrath first author of ch. 6, last author of ch. 2 | Failed (wording) |
| Australian guide, 2022, National eDNA Reference Centre | CSIRO PDF | yes | "Suter et al. 2022" | "De Brauwer M, Chariton A, ... Suter L, ... Trujillo-Gonzalez A (2022) ... National eDNA Reference Centre, Canberra" | Failed (author) |
| Natural England WC1067 (2014) | SureScreen page | yes (name); year not on page | WC1067, 2014 | Page names WC1067 as "the regulated method by Natural England"; 2014 not on any page opened | Held (year not checked) |
| SureScreen "strictly follows WC1067" | SureScreen page | yes | quote | "Our analysis strictly follows WC1067" | Held |
| CSA W214:21 (2021) | CSA store | could not open (404) | search only | — | Not checked (as stated) |
| CEN EN 17805:2023, withdrawn 15 Jul 2026 | CEN, iTeh, BSI | could not open | search only | — | Not checked (as stated) |
| 5 UK agencies, Water Framework Directive | Hull REF case study | yes | EA, SEPA, NRW, NE, NatureScot | Exactly these five, monitoring under the EU Water Framework Directive | Held |
| "first example ... statutory monitoring" | Hull REF | yes | quote | Exact quote present (the university's own claim) | Held |
| UK lakes use 12S fish metabarcoding | Hull REF | yes | "12S fish metabarcoding (UK lakes)" | "eDNA metabarcoding of fish communities"; "12S" appears 0 times | Failed (not in source) |
| SureScreen GBP 140/180/240/380 | SureScreen page | yes | 10/5/2/same day | £140, £180, £240, £380 +VAT per analysis | Held |
| SureScreen kit GBP 30 | SureScreen page | yes | search only | "GCN eDNA Kit – £30 +VAT" | Held (now opened) |
| SureScreen second species GBP 35 | SureScreen page | yes | search only | "For an extra £35 +VAT, you can add ..." | Held (now opened) |
| Cellmark GBP 170/220/270 per kit | Cellmark page | yes | 10/5/2 days | "10 day £170, 5 day £220 and 2 day £270 +VAT" | Held |
| Cellmark from GBP 155 for 20+ kits; next day | Cellmark page | yes | as stated | "from £155 ... For orders of 20 kits or more"; Next Day listed | Held |
| Wilderlab "tens of thousands of samples per year" | Farmers Weekly, 20 Jun 2023 | yes | quote, for councils | "We do tens of thousands of samples per year for them" (local government) | Held |
| Wilderlab USD 114 basic / USD 182 comprehensive | wilderlab.co/extracts, /directions | yes | per extract | Basic NZD 192 / USD 114; Comprehensive NZD 307 / USD 182 | Held |
| Price step 182 - 114 | python | — | USD 68 | 68 (NZD 115; GBP 56) | Held |
| USD 68 as a price for a COI pass | directions; methods PDF | yes | price anchor | COI assay CI is in the Basic panel; the 68 buys plant, microbe, 18S, ITS assays | Failed (as a COI anchor) |
| Wilderlab assay list | directions | yes | 12S, 16S, COI, 18S-V9, 16S-V5, ITS, trnL, 28S, D-loop | All present. Methods v2.1.0 also lists rbcL; the current table does not | Held |
| Wilderlab classification steps | methods v2.1.0 | yes | exact match, LCA on GenBank/BOLD/RDP, SINTAX | Four steps; step 3 (one mismatch allowed) was missing; RDP only for the UM assay | Failed |
| "genus at most for 16S and 18S" | methods PDF | yes | 16S and 18S | Cap is for assays BE, BU, UM (18S-V9, bacterial 16S-V5), not mitochondrial 16S | Failed (too broad) |
| Wilkinson wrote `insect`, the classifier used | methods; bio | yes | as stated | `insect` is used to demultiplex; naming is exact match, LCA, SINTAX | Failed |
| Wilkinson founder and CEO | conference bio | yes | as stated | "founder and CEO of Wilderlab"; founded 2019 | Held |
| "over 50,000" samples, 9 Oct 2024 | conference bio | yes | running total | "having processed over 50,000 samples"; dated October 9, 2024 | Held |
| "over 23,000" samples, undated | AIM2Flourish | yes | search only | "over 23,000 eDNA samples"; undated | Held (now opened) |
| Running totals "agree in size" with 20,000 a year | python | — | agree | 50,000 over 5.77 years = 8,663 a year | Failed (wording) |
| NatureMetrics "the species present", "proprietary" | naturemetrics.com/gcn | yes | quotes | Both quotes present; no prices | Held |
| Jonah Ventures 8 awards, USD 13K-106K | govconinabox | yes | 8 awards | 29 awards, USD 814K total; 8 name eDNA, USD 13K-106K | Held (count is a selection) |
| USD 64K for 3,840 libraries, Commerce, Sep 2024 | govconinabox | yes | as stated | Matches, 16 Sep 2024 | Held |
| USD 17 per library | python | — | about 17 | 64,000 / 3,840 = 16.67 | Held |
| Genidaqs units only | rate list | yes | units, no amounts | Matches; 48 h qPCR option | Held |
| TAMUCC 23.63 / 120.80 / 14.13 / 115 per h, FY 2027-2028 | TAMUCC page | yes | as stated | Matches | Held |
| UMaine 13 / 35 / 450 | UMaine iLab | yes | as stated | Matches | Held |
| SPYGEN co-authors Jaquier 2024 | Europe PMC | yes | co-author | Valentini A and Dejean T, SPYGEN | Held |
| Jaquier: ecotag against EMBL | Europe PMC | yes | as stated | "ecotag ... EMBL ... v141"; 100 / 90-99 / 85-90% identity | Held |
| Providers: 2 read, 2 qPCR, 5 not found, of 9 | report | — | 9 | 2 + 2 + 5 = 9 | Held |
| Teixeira 87% of 5,375 ASVs | PMC12178944 | yes | 55 + 12 + 20 | 55 + 12 + 20 = 87; 679 at genus = 12.6% | Held |
| Teixeira: annelid ASVs from reef plates | Europe PMC | yes | annelid, ARMS | 135 reef monitoring structures; filtered to Annelida; not water eDNA | Held (not water) |
| Teixeira Leray 313 bp, RDP | Europe PMC | yes | as stated | Matches | Held |
| Jeunen 574 of 1,450, at least 39.6% | LJMU PDF | yes | 574/1,450 | "574 OTUs (40%)"; 0.3959. Results text gives 574 as 35.45% of another base | Held |
| Jeunen: BLAST, CRABS, LCA | LJMU PDF | yes | as stated | Matches (Wilkinson is a co-author) | Held |
| Cerrillo-Espinosa 1,694 of 5,495, at least 30.8% | PMC12024446 | yes | "above phylum" | Matches; also counts unresolved SAR; 3,229 no-hit OTUs dropped earlier | Held (SAR part omitted) |
| Cerrillo 0 to over 200 m; 313 bp | Europe PMC | yes | as stated | Matches | Held |
| Saenz-Agudelo "at least 8% to 38%" | PMC10881902 | yes | phylum range | Table 3 gives genus: COI 25.6%, 18S 46.4%, 16S 85.2% named, so 74.4 / 53.6 / 14.8% not | Failed |
| Saenz TSV classifier "not read", n "not reported" | Europe PMC | yes | not read | Anacapa: Bowtie2 + BCLA, 0.8; ASV counts in Table 3 | Failed |
| Jaquier 17.0% of 264 fish taxa | PMC11099785 | yes | 45 | 264 - 141 - 78 = 45; 0.1705 | Held |
| "four of the five as lower bounds" | TSV | — | four | three; two after the Saenz fix | Failed |
| "17% to 87%" | TSV | — | 17 to 87 | 14.8% to 87% after the Saenz fix | Failed |
| Step 3 rule outcome | TSV | — | no kill | Still no kill | Held |
| "The gap is largest in COI" | Teixeira; Saenz Table 3 | yes | COI largest | Teixeira 87%; Saenz COI 74.4 > 18S 53.6 > 16S 14.8 | Held |
| 313 bp is about 104 codons | — | — | 104 | 313 / 3 = 104.3 | Held |
| Gómez-Repollés 0.78% of reads, BLASTn vs nt | Pensoft | yes | 0.78% | Exact quote; 15 samples; 18 Feb 2026 | Held |
| Prokaryote non-coding DNA 5% to 50%, 50 genomes | BioNumbers 105749 | yes | 5-50% | Matches; under 18% in 90% of genomes | Held |
| Human exome about 1.5% | NHGRI | yes | 1.5% | Exact quote | Held |
| Norway 995 seawater sites, 31 Dec 2025 | Fiskeridirektoratet Table 9 | yes | 995 | County sum 83+115+221+158+79+270+59+10+0 = 995 | Held |
| PatoGen: real-time PCR, Ålesund | medical-xprt | yes | search only | Matches | Held (now opened) |
| PondNet 380 ponds; since 2015; Defra and NE since 2018 | Freshwater Habitats Trust | yes | as stated | Matches | Held |
| Samples for USD 50,000 | python | — | 736 | 50,000 / 68 = 735.29; 736 x 68 = 50,048 | Held |
| Share of 20,000 | python | — | 3.7% | 0.0368 | Held |
| All 20,000 buy | python | — | 1,360,000 | 1,360,000 | Held |
| At 15,000 a year | python | — | 4.9% | 0.04907 | Held |
| At the running-total average | python | — | not in draft | 736 / 8,663 = 8.5% | (added) |
| Capital Mag "around 15,000 a year" | capitalmag.co.nz | could not open | not used | — | Not checked (as stated) |
| East Anglia TWO Appendix 22.4 | — | not tried | search only | — | Not checked (as stated) |
| Branchwater / sourmash top contributors | GitHub API | yes | Irber, Brown | branchwater: luizirber 61, bluegenes 17, ctb 6; sourmash: ctb 1,219, luizirber 296 | Held |
| Step 5 decision follows the rule | rule text | — | "It does" | Shows a ceiling, not uptake; 68 is a panel step, not an add-on, and buys no COI | Held only under a loose reading |

## What was done with each correction

All 17 suggested corrections were applied to the report and data file on 2026-10-18,
after the report's author reopened three of them (the guide's citation page, Wilderlab's
methods Table 1, and Saenz-Agudelo Table 3) and saw the same thing:

| Correction | Applied |
|---|---|
| De Brauwer et al. 2022 | yes |
| Step 3 bullet: 14.8% to 87%, two lower bounds | yes |
| Saenz-Agudelo row split into COI, 18S, 16S from Table 3; figure redrawn | yes |
| Saenz classifier and ASV counts in the TSV | yes |
| Cerrillo-Espinosa row mentions SAR | yes |
| Wilderlab four classification steps; genus cap only for 18S-V9 and 16S-V5 | yes |
| Wilkinson row: `insect` sorts reads by barcode | yes |
| Bockrath row: listed first on the FWS page; first author of chapter 6 | yes |
| UK lakes: marker not named | yes |
| Step 5: USD 68 is a panel step, not an add-on, and buys no COI; both throughput readings (20,000 and 8,663 a year); uptake assumed, not shown | yes |
| Every Wilderlab panel includes COI | yes |
| Items now opened moved to "Checked" | yes |
| Section 3 says four studies are seawater, one reef plates | yes |
| Jonah Ventures: 29 awards, USD 814K, 8 name eDNA | yes |

The decision stays "worth a conversation" under the prompt's own step 5 method (price
times one provider's sample count), and the report now also says plainly that, read
strictly, the rule is not yet met: there is no add-on price and no evidence of uptake.
