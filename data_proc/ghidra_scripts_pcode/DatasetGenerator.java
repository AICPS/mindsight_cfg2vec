//AST Generator, this is for generating pcode along with acfgs
//@author Shih-Yuan Yu
//@category MINDSIGHT
import java.util.ArrayList;
import java.util.Iterator;
import java.util.HashMap;
import java.util.*;
import java.util.Map;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Program;
import ghidra.program.model.listing.StackFrame;
import ghidra.program.model.listing.*;
import ghidra.program.model.symbol.*;
import ghidra.program.model.block.*;
import ghidra.program.model.data.StringDataInstance;
import ghidra.program.model.pcode.*;
import ghidra.program.model.address.*;
import ghidra.program.util.DefinedDataIterator;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.decompiler.ClangNode;
import ghidra.util.task.TaskMonitor;


class Tuple<X, Y> { 
	public final X x; 
	public final Y y; 
	public Tuple(X x, Y y) { 
	  this.x = x; 
	  this.y = y; 
	} 
} 


class BasicBlockMetadata<X, Y, Z, W> {
	ArrayList<X> basicBlocks;
    ArrayList<Y> edges;
    ArrayList<Z> attributes;
    ArrayList<W> boundaries;

	public BasicBlockMetadata (ArrayList<X> basicBlocks, ArrayList<Y> edges, ArrayList<Z> attributes, ArrayList<W> boundaries) {
		this.basicBlocks = basicBlocks;
        this.edges = edges;
        this.attributes = attributes;
        this.boundaries = boundaries;
	}

    @Override
    public String toString() { 
        return "BasicBlockMetdata toString() to be implemented";
    } 
}


class Utils {
    public static String getPackageName(String path){
    	System.out.println("Package path: " + path);
		File file = new File(path);
        return file.getName();
    }
	public static DecompInterface setUpDecompiler(Program program) {
		DecompInterface decompilerInterface = new DecompInterface();

		// call it to get results
		if (!decompilerInterface.openProgram(program)) {
			System.out.print("Decompile Error: " + decompilerInterface.getLastMessage());
			return null;
		}

		DecompileOptions options;
		options = new DecompileOptions();
		decompilerInterface.setOptions(options);

		decompilerInterface.toggleCCode(true);
		decompilerInterface.toggleSyntaxTree(true);
		decompilerInterface.setSimplificationStyle("decompile");

		return decompilerInterface;
	}

	public static void createDir(String dirName) {
		File directory = new File(dirName);
		if (!directory.exists()) {
			directory.mkdirs();
		}
	}
}


public class DatasetGenerator extends GhidraScript {

	// PCode Description: https://ghidra.re/courses/languages/html/pcoderef.html4
	// Other interesting links and doc
	// https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/main/help/help/topics/DecompilePlugin/Decompiler.htm
	// Pcode Trees:
	// https://ghidra.re/courses/languages/html/sleigh_constructors.html#idm140310874886224
	// Scripts that dump Pcode data:
	// https://github.com/d-millar/ghidra_pcode_scripts

	private DecompInterface decompInterface;

	@Override
	public void run() throws Exception {
		String acfgOutputDirBase = getScriptArgs()[0]; // assuming that it should have at least one arugment specifying the storing folder.
		Path acfgOutputDir = Paths.get(acfgOutputDirBase, Utils.getPackageName(currentProgram.getExecutablePath()), currentProgram.getName() + "-acfg/");
		Path ccodeDir = Paths.get(acfgOutputDirBase, Utils.getPackageName(currentProgram.getExecutablePath()), currentProgram.getName() + "-ccode/");
		String ccodeoutdir = ccodeDir.toAbsolutePath().toString();
		String outputdir = acfgOutputDir.toAbsolutePath().toString();
		Utils.createDir(outputdir);
		Utils.createDir(ccodeoutdir);
		println(outputdir);

		decompInterface = Utils.setUpDecompiler(currentProgram);

		long startTime = System.currentTimeMillis();
		generate_pcode_callgraph(outputdir, ccodeoutdir);
		println("Elapsed time: " + (System.currentTimeMillis() - startTime));

		decompInterface.dispose();
	}
	
	private void generate_pcode_callgraph(String outDir, String ccodeoutDir) throws Exception {
		System.out.println("Generating PCode & CallGraph ...");
		
		FileWriter codeWriter = new FileWriter(Paths.get(outDir, "pcode.csv").toAbsolutePath().toString());
		FileWriter callGraphWriter = new FileWriter(Paths.get(outDir, "callgraph.csv").toAbsolutePath().toString());
		FileWriter xRefWriter = new FileWriter(Paths.get(outDir, "xrefs.csv").toAbsolutePath().toString());
		FileWriter functionBoundaryWriter = new FileWriter(Paths.get(outDir, "function_boundaries.csv").toAbsolutePath().toString());
		FileWriter edgeWriter = new FileWriter(Paths.get(outDir, "edges.csv").toAbsolutePath().toString());
		FileWriter attributeWriter = new FileWriter(Paths.get(outDir, "attributes.csv").toAbsolutePath().toString());
		FileWriter boundaryWriter = new FileWriter(Paths.get(outDir, "block_boundaries.csv").toAbsolutePath().toString());
		FileWriter bbcodeWriter = new FileWriter(Paths.get(outDir, "bb_pcode.csv").toAbsolutePath().toString());
		System.out.println("Generating ACFG ...");
		for (Function function: currentProgram.getFunctionManager().getFunctions(true)) { // true means the functions will be ordered ascendingly.
			/* extracting pcode for every functions */
			Listing listing = currentProgram.getListing();
			InstructionIterator opiter = listing.getInstructions(function.getBody(), true);
			while (opiter.hasNext()) {
				Instruction insn = opiter.next();
				PcodeOp[] raw_pcode = insn.getPcode();
				for (int i = 0; i < raw_pcode.length; i++) {
					codeWriter.write(function.getName() + ", " + raw_pcode[i].toString() + '\n');
				}
			}
						
			/* extracting function boundaries for every function */ 
			functionBoundaryWriter.write(function.getName() + ", " + function.getEntryPoint().toString() + "\n");

			/* extracting calling relations and xrefs for non-thunk functions only! */
			if (function.isThunk() == false) {			
				for (Function outgoing_function : function.getCalledFunctions(ghidra.util.task.TaskMonitor.DUMMY)) {
					callGraphWriter.write("outgoing, " + function.getName() + ", " + outgoing_function.getName() + "\n");
				}

				Reference[] references = currentProgram.getReferenceManager().getFlowReferencesFrom(function.getEntryPoint());
				for (int i = 0; i < references.length; i++) {
					xRefWriter.write("flowReferenceFrom, " + function.getName() + ", " + references[i].toString() + "\n");
				}
			}
			/* anything below here needs decompiling the function */
			/* extracting decompiled ccode for every function */
			DecompileResults results = decompInterface.decompileFunction(function, decompInterface.getOptions().getDefaultTimeout(), TaskMonitor.DUMMY);
			ArrayList<PcodeBlockBasic> basicBlocks = new ArrayList<>();
			ArrayList<Tuple<String, String>> edges = new ArrayList<>();
			ArrayList<String> attributes = new ArrayList<>();
			ArrayList<String> boundaries = new ArrayList<>();
			BasicBlockMetadata<PcodeBlockBasic, Tuple<String, String>, String, String> bbMetadata = null;
			if (results.failedToStart()) {
				System.out.println("Failed to start decompilation");
			}
			if (!results.decompileCompleted()) {
				System.out.println(results.getErrorMessage());
			} else {
				HighFunction hf = results.getHighFunction();
				String function_ccode_path = Paths.get(ccodeoutDir, function.getName() + ".txt").toAbsolutePath().toString();
				if (function.isThunk() == true)
					function_ccode_path = Paths.get(ccodeoutDir, function.getName() + "_thunk.txt").toAbsolutePath().toString();
				FileWriter ccodeWriter = new FileWriter(function_ccode_path);
				ccodeWriter.write(results.getDecompiledFunction().getC());
				ccodeWriter.close();
				
				if (function.isThunk() == true) {
					continue;
				}
				
				ArrayList<PcodeBlockBasic> bbs = hf.getBasicBlocks();

				Map<Long, String> programStrings = getProgramStrings();

				// https://ghidra.re/ghidra_docs/api/ghidra/program/model/pcode/PcodeBlockBasic.html
				for (PcodeBlockBasic bb : bbs) {
					java.util.Iterator<PcodeOp> pcodeIter = bb.getIterator();
					try {
						while (pcodeIter.hasNext()) {
							PcodeOp pcode = pcodeIter.next();
							bbcodeWriter.write(function.getName() + ", " + bb.toString() + ", " + pcode.toString() + '\n');
						}
					} catch (Exception e) {
						e.printStackTrace();
					}
					Map<String, Number> bbFeatures = getBasicBlockFeatures(bb, programStrings);
					basicBlocks.add(bb);
					attributes.add(function.getName() + "," + bb.toString() + "," + printBasicBlockFeatures(bbFeatures, false));

					// Process Input BBs
					for (int i = 0; i < bb.getInSize(); i++) {
						String src = bb.getIn(i).toString();
						String dst = bb.toString();
						edges.add(new Tuple<String, String>(src, dst));
					}

					boundaries.add(bb.getStart().toString() + ", " + bb.getStop().toString());
				}
			}
			bbMetadata = new BasicBlockMetadata<PcodeBlockBasic, Tuple<String, String>, String, String>(basicBlocks, edges, attributes, boundaries);

			// Print Program (function-basic blocks)
			for (int i = 0; i < bbMetadata.basicBlocks.size(); i++) {
				PcodeBlockBasic bb = (PcodeBlockBasic) bbMetadata.basicBlocks.get(i);

				// Print block boundaries
				String boundary = (String) bbMetadata.boundaries.get(i);
				boundaryWriter.write(function.getName() + ", " + bb.toString() + ", " + boundary + "\n");
			}
			// Print Edges (bb-bb)
			for (int i = 0; i < bbMetadata.edges.size(); i++) {
				Tuple edge = (Tuple) bbMetadata.edges.get(i);
				edgeWriter.write(function.getName() + ", " + edge.x + ", " + edge.y + "\n");
			}
			// Print attributes
			for (int i = 0; i < bbMetadata.attributes.size(); i++) {
				String attribute = (String) bbMetadata.attributes.get(i);
				attributeWriter.write(attribute + "\n");
			}
		}

		edgeWriter.close();
		attributeWriter.close();
		boundaryWriter.close();
		bbcodeWriter.close();
		codeWriter.close();
		callGraphWriter.close();
		xRefWriter.close();
		functionBoundaryWriter.close();
		System.out.println("Finished - analyzing PCode & CallGraph for " + currentProgram.getName());
	}

	// Get all the strings in the program
	// XXX: the Address from definedStrings and varnode do not match. One gives the
	// address 0x1111 the other gives (ram)0x1111 and the match finds.
	// XXX: keeping it as Long
	private Map<Long, String> getProgramStrings() {
		Map<Long, String> programStrings = new HashMap<>();
		for (Data data : DefinedDataIterator.definedStrings(currentProgram)) {
			StringDataInstance str = StringDataInstance.getStringDataInstance(data);
			if (StringDataInstance.NULL_INSTANCE == str) {
				continue;
			}
			programStrings.put(str.getAddress().getOffset(), str.getStringValue());
		}
		return programStrings;
	}

	private PcodeBlockBasic getInstructionBasicBlock(Instruction insn, List<PcodeBlockBasic> bbs) {
		for (PcodeBlockBasic bb: bbs) {
			if (insn.getFallFrom() == null)
				return bbs.get(0);
			if (bb.contains(insn.getFallFrom()))
				return bb;
		}
		return null;
	}
	
	private ArrayList<PcodeBlockBasic> getBasicBlocks(Function function) {
		ArrayList<PcodeBlockBasic> bbList = new ArrayList<>();
		DecompileResults results = decompInterface.decompileFunction(function,
				decompInterface.getOptions().getDefaultTimeout(), TaskMonitor.DUMMY);
		if (results.failedToStart()) {
			System.out.println("Failed to start decompilation");
		}

		if (!results.decompileCompleted()) {
			System.out.println(results.getErrorMessage());
		} else {
			HighFunction hf = results.getHighFunction();
			ArrayList<PcodeBlockBasic> bbs = hf.getBasicBlocks();
			for (PcodeBlockBasic bb : bbs) {
				java.util.Iterator<PcodeOp> pcodeIter = bb.getIterator();
				bbList.add(bb);
			}
		}
		return bbList;
	}

	// Get features for a basic block instructions
	private Map<String, Number> getBasicBlockFeatures(PcodeBlockBasic bb, Map<Long, String> programStrings) {
		Iterator<PcodeOp> insns = bb.getIterator();
		Map<String, Number> features = new HashMap<>();

		// Features: https://github.com/qian-feng/Gencoding/blob/master/raw-feature-extractor/graph_analysis_ida.py
		int totalInstructions = 0;
		int arithmeticInstructions = 0;
		int logicInstructions = 0;
		int transferInstructions = 0;
		int callInstructions = 0;
		int otherInstructions = 0;
		int dataTransferInstructions = 0;
		int ssaInstructions = 0;
		int pointerInstructions = 0;
		int compareInstructions = 0;
		int totalConstants = 0;
		int totalStrings = 0;

		while (insns.hasNext()) {
			PcodeOp node = insns.next();
			String mnemonic = node.getMnemonic();
			switch (mnemonic) {
				case "INT_ADD":
				case "INT_SUB":
				case "INT_MULT":
				case "INT_DIV":
				case "INT_REM":
				case "INT_SDIV":
				case "INT_SREM":
				case "INT_LEFT":
				case "INT_RIGHT":
				case "INT_SRIGHT":
				case "FLOAT_ADD":
				case "FLOAT_SUB":
				case "FLOAT_MULT":
				case "FLOAT_DIV":
				case "FLOAT_ABS":
				case "FLOAT_SQRT":
				case "FLOAT_CEIL":
				case "FLOAT_FLOOR":
					// Not sure about these belong here
				case "INT_ZEXT":
				case "INT_SEXT":
					arithmeticInstructions += 1;
					break;

				case "BOOL_NEGATE":
				case "BOOL_AND":
				case "BOOL_XOR":
				case "BOOL_OR":
					// Not sure about these belong here
				case "INT_OR":
				case "INT_AND":
				case "INT_XOR":
					logicInstructions += 1;
					break;

				case "BRANCH":
				case "CBRANCH":
				case "BRANCHIND":
				case "RETURN":
					transferInstructions += 1;
					break;

				case "CALL":
				case "CALLIND":
					callInstructions += 1;
					break;

				case "COPY":
				case "LOAD":
				case "STORE":
					// Not sure about these belong here
				case "PIECE":
				case "SUBPIECE":
				case "CAST":
					dataTransferInstructions += 1;
					break;

				case "MULTIEQUAL":
				case "INDIRECT":
					ssaInstructions += 1;
					break;

				case "INT_EQUAL":
				case "INT_NOTEQUAL":
				case "INT_LESS":
				case "INT_SLESS":
				case "INT_LESSEQUAL":
				case "INT_SLESSEQUAL":
				case "FLOAT_EQUAL":
				case "FLOAT_NOTEQUAL":
				case "FLOAT_LESS":
				case "FLOAT_LESSEQUAL":
					compareInstructions += 1;
					break;

				case "PTRSUB":
				case "PTRADD":
					pointerInstructions += 1;
					break;

				default:
					// println(mnemonic);
					otherInstructions += 1;
					break;
			}

			for (int i = 0; i < node.getNumInputs(); i++) {
				Varnode operand = node.getInput(i);
				if (operand.isConstant()) {
					totalConstants += 1;
				}
				if (operand.toString() != null) {
					Address myAddr = operand.getAddress();
					if (programStrings.containsKey(myAddr.getOffset())) {
						totalStrings += 1;
					}
				}

			}

			totalInstructions += 1;
		}

		features.put("totalInstructions", totalInstructions);
		features.put("arithmeticInstructions", arithmeticInstructions);
		features.put("logicInstructions", logicInstructions);
		features.put("transferInstructions", transferInstructions);
		features.put("callInstructions", callInstructions);
		features.put("dataTransferInstructions", dataTransferInstructions);
		features.put("ssaInstructions", ssaInstructions);
		features.put("compareInstructions", compareInstructions);
		features.put("pointerInstructions", pointerInstructions);
		features.put("otherInstructions", otherInstructions);
		features.put("totalConstants", totalConstants);
		features.put("totalStrings", totalStrings);
		return features;
	}

	private String printBasicBlockFeatures(Map<String, Number> features, Boolean dotFormat) {
		StringBuilder myString = new StringBuilder(" ");
		if (dotFormat == true) 
			myString.append("\"[");
		myString.append(features.get("totalInstructions"));
		myString.append(", ");
		myString.append(features.get("arithmeticInstructions"));
		myString.append(", ");
		myString.append(features.get("logicInstructions"));
		myString.append(", ");
		myString.append(features.get("transferInstructions"));
		myString.append(", ");
		myString.append(features.get("callInstructions"));
		myString.append(", ");
		myString.append(features.get("dataTransferInstructions"));
		myString.append(", ");
		myString.append(features.get("ssaInstructions"));
		myString.append(", ");
		myString.append(features.get("compareInstructions"));
		myString.append(", ");
		myString.append(features.get("pointerInstructions"));
		myString.append(", ");
		myString.append(features.get("otherInstructions"));
		myString.append(", ");
		myString.append(features.get("totalConstants"));
		myString.append(", ");
		myString.append(features.get("totalStrings"));
		if (dotFormat == true) 
			myString.append("]\"");
		return myString.toString();
	}
}