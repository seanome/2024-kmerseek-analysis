import groovy.json.JsonOutput

// Run with the pipeline's pinned Nextflow jar, which supplies its actual Groovy runtime.
def helper = new GroovyClassLoader().parseClass(new File('lib/FitCache.groovy'))
def first = helper.indexSuffix('1.59', '6.36', 500, 'image-a')
assert first != helper.indexSuffix('1.59', '12.72', 500, 'image-a')
assert first != helper.indexSuffix('1.59', '6.36', 1000, 'image-a')
assert first != helper.indexSuffix('1.59', '6.36', 500, 'image-b')
assert helper.indexSuffix('', '', 500, 'image-a') == ''
def name = "human.hp_pbotc_1st_ed2.k19.lcfalse${first}.kmerseek.rocksdb"
def parser = new File('main.nf').readLines().find { it.contains('d.name =~') && it.contains('rocksdb') }
assert parser != null
def match = name =~ parser.substring(parser.indexOf('/') + 1, parser.lastIndexOf('/'))
assert match && match[0][1..4] == ['human', 'hp_pbotc_1st_ed2', '19', 'false']

def directory = java.nio.file.Files.createTempDirectory('fit-cache-test')
def stem = 'human_vs_human.hp_pbotc_1st_ed2.k19.lcfalse'
def settings = helper.searchSettings(name, 0.01, 2, 0.05, 1.3)
try {
    helper.checkSearch(directory, stem, settings) // A new output directory is valid.
    directory.resolve(stem + '.regions.parquet').toFile().text = 'cached result'
    def refused = false
    try { helper.checkSearch(directory, stem, settings) }
    catch (IllegalStateException expected) { refused = true }
    assert refused // A legacy file without provenance must not silently satisfy a run.
    directory.resolve(stem + '.search-settings.json').toFile().text = JsonOutput.toJson(settings)
    helper.checkSearch(directory, stem, settings)
    def changed = helper.searchSettings(name, 0.01, 2, 0.05, 2.0)
    refused = false
    try { helper.checkSearch(directory, stem, changed) }
    catch (IllegalStateException expected) { refused = true }
    assert refused
} finally {
    directory.toFile().deleteDir()
}
println('Cache identity and stale-result checks passed')
