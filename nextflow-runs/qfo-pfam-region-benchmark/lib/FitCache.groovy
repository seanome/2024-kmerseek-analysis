import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import java.nio.file.Path
import java.security.MessageDigest

/** Keep fitted indexes and cached searches tied to their settings and image reference. */
class FitCache {
    static String indexSuffix(String penalty, String xdrop, int queries, String image) {
        if (!penalty) return ''
        def digest = MessageDigest.getInstance('SHA-256')
            .digest(image.getBytes('UTF-8')).encodeHex().toString()
        return ".ka-c${penalty}.x${xdrop}.q${queries}.i${digest}"
    }

    static Map searchSettings(String indexName, threshold, minShared, maxPvalue, minScore) {
        return [index: indexName, threshold: threshold.toString(),
                min_shared_kmers: minShared.toString(),
                max_query_pvalue: maxPvalue.toString(), min_region_score: minScore.toString()]
    }

    static void checkSearch(Path directory, String stem, Map settings) {
        if (!directory.resolve(stem + '.regions.parquet').toFile().exists()) return
        def manifest = directory.resolve(stem + '.search-settings.json').toFile()
        if (!manifest.exists() || new JsonSlurper().parse(manifest) != settings) {
            throw new IllegalStateException(
                "Cached extended search ${stem} has missing or different settings. " +
                'Use a new --outdir; preserve the old results as evidence.')
        }
    }

    static String json(Map settings) {
        return JsonOutput.toJson(settings)
    }
}
