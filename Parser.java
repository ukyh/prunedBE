import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.PropertiesUtils;

import java.util.*;
import java.io.*;


public class Parser {

    public static void main(String[] args) throws IOException {

        String dataset = args[0];
        // Dependency and annotation are fixed at "ud" and "++" in pBE
        // We just leave these options for future use
        String dep = "ud";    // "ud" or "sd"
        String anno = "++";    // "++" or "basic"

        // build pipeline
        StanfordCoreNLP pipeline = null;
        if (dep.equals("ud")) {
            pipeline = new StanfordCoreNLP(
                        PropertiesUtils.asProperties(
                        "annotators", "tokenize,ssplit,pos,depparse",
                        "depparse.model", "edu/stanford/nlp/models/parser/nndep/english_UD.gz",
                        "tokenize.language", "en"));
        } else if (dep.equals("sd")) {
            pipeline = new StanfordCoreNLP(
                        PropertiesUtils.asProperties(
                        "annotators", "tokenize,ssplit,pos,depparse",
                        "depparse.model", "edu/stanford/nlp/models/parser/nndep/english_SD.gz",
                        "tokenize.language", "en"));
        }

        // get reference file names
        String refPath = dataset + "/ref/";
        File refFile = new File(refPath);
        String[] refFiles = refFile.list();
        List<String> refNames = new ArrayList<String>(Arrays.asList(refFiles));

        // iterate each reference file
        for (String name : refNames) {
            // exclude hidden files
            if (!name.startsWith(".")) {
                // make output file
                File parsedFile = new File(dataset + "/ref_parsed/" + name);
                PrintStream outputPrintStream = new PrintStream(parsedFile);
                System.setOut(outputPrintStream);

                // iterate each line
                try(BufferedReader br = new BufferedReader(new FileReader(refPath + name))) {
                    for(String line; (line = br.readLine()) != null; ) {
                        // annotate line
                        Annotation document = new Annotation(line);
                        pipeline.annotate(document);

                        // get the dependecy annotation
                        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
                        for(CoreMap sentence: sentences) {
                            if (anno.equals("++")) {
                                SemanticGraph dependencies = sentence.get(EnhancedPlusPlusDependenciesAnnotation.class);
                                System.out.println(dependencies.toList());
                            } else if (anno.equals("basic")) {
                                SemanticGraph dependencies = sentence.get(BasicDependenciesAnnotation.class);
                                System.out.println(dependencies.toList());
                            }
                        }
                    }
                }
            } 
        }

        System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out)));

        // get target file names
        String trgPath = dataset + "/trg/";
        File trgFile = new File(trgPath);
        String[] trgFiles = trgFile.list();
        List<String> trgNames = new ArrayList<String>(Arrays.asList(trgFiles));

        // iterate each target file
        for (String name : trgNames) {
            // exclude hidden files
            if (!name.startsWith(".")) {
                // make output file
                File parsedFile = new File(dataset + "/trg_parsed/" + name);
                PrintStream outputPrintStream = new PrintStream(parsedFile);
                System.setOut(outputPrintStream);

                // iterate each line
                try(BufferedReader br = new BufferedReader(new FileReader(trgPath + name))) {
                    for(String line; (line = br.readLine()) != null; ) {
                        // annotate line
                        Annotation document = new Annotation(line);
                        pipeline.annotate(document);

                        // get the dependecy annotation
                        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
                        for(CoreMap sentence: sentences) {
                            if (anno.equals("++")) {
                                SemanticGraph dependencies = sentence.get(EnhancedPlusPlusDependenciesAnnotation.class);
                                System.out.println(dependencies.toList());
                            } else if (anno.equals("basic")) {
                                SemanticGraph dependencies = sentence.get(BasicDependenciesAnnotation.class);
                                System.out.println(dependencies.toList());
                            }
                        }
                    }
                }
            }
        }
    }
}
