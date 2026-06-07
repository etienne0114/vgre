#!/usr/bin/env node
/**
 * VGRE Local MCP Verification Server
 *
 * Provides tools for automated verification of the VGRE codebase:
 *   - build            : cmake --build build -j<N>
 *   - test             : ctest in build directory
 *   - scan_stubs       : grep for TODOs, stubs, hardcoded magic values
 *   - scan_crossplatform: detect POSIX-only calls without guards
 *   - validate_symbol  : check if a specific API function is implemented
 *   - check_missing    : compare declared headers vs implemented symbols
 *
 * Protocol: MCP over stdio (JSON-RPC 2.0 framing with Content-Length header).
 */

'use strict';

const { execSync, spawnSync } = require('child_process');
const path  = require('path');
const fs    = require('fs');
const os    = require('os');

const PROJECT_ROOT = process.env.VGRE_ROOT ||
    path.resolve(__dirname, '..', '..');
const BUILD_DIR   = path.join(PROJECT_ROOT, 'build');
const CPU_COUNT   = os.cpus().length;

// ─── MCP stdio transport ─────────────────────────────────────────────────────
// Reads Content-Length framed JSON-RPC messages from stdin; writes to stdout.

let inputBuf = Buffer.alloc(0);

process.stdin.on('data', (chunk) => {
    inputBuf = Buffer.concat([inputBuf, chunk]);
    processMessages();
});
process.stdin.on('end', () => process.exit(0));

function processMessages() {
    while (true) {
        // Find the header / body separator
        const sep = inputBuf.indexOf('\r\n\r\n');
        if (sep === -1) return; // incomplete header
        const header = inputBuf.slice(0, sep).toString('utf8');
        const lenMatch = header.match(/Content-Length:\s*(\d+)/i);
        if (!lenMatch) { inputBuf = inputBuf.slice(sep + 4); continue; }
        const bodyLen = parseInt(lenMatch[1], 10);
        if (inputBuf.length < sep + 4 + bodyLen) return; // incomplete body
        const body = inputBuf.slice(sep + 4, sep + 4 + bodyLen).toString('utf8');
        inputBuf = inputBuf.slice(sep + 4 + bodyLen);
        try { handleMessage(JSON.parse(body)); } catch (e) { /* ignore parse errors */ }
    }
}

function sendMessage(obj) {
    const body = JSON.stringify(obj);
    const frame = `Content-Type: application/json\r\nContent-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`;
    process.stdout.write(frame);
}

function sendResult(id, result) {
    sendMessage({ jsonrpc: '2.0', id, result });
}
function sendError(id, code, message) {
    sendMessage({ jsonrpc: '2.0', id, error: { code, message } });
}

// ─── Tool definitions ────────────────────────────────────────────────────────

const TOOLS = [
    {
        name: 'build',
        description: 'Build the VGRE project with cmake. Reports errors and warnings.',
        inputSchema: {
            type: 'object',
            properties: {
                target: { type: 'string', description: 'Optional cmake target (default: all)', default: '' },
                jobs:   { type: 'integer', description: 'Parallel jobs (default: CPU count)', default: CPU_COUNT }
            },
            required: []
        }
    },
    {
        name: 'test',
        description: 'Run the full VGRE test suite with ctest. Returns pass/fail summary.',
        inputSchema: {
            type: 'object',
            properties: {
                filter:  { type: 'string', description: 'Regex to filter test names (optional)' },
                timeout: { type: 'integer', description: 'Per-test timeout in seconds (default: 120)', default: 120 }
            },
            required: []
        }
    },
    {
        name: 'scan_stubs',
        description: 'Scan the codebase for stubs, TODOs, hardcoded magic values, and missing implementations.',
        inputSchema: {
            type: 'object',
            properties: {
                path: { type: 'string', description: 'Subdirectory to scan (default: src/api)', default: 'src/api' }
            },
            required: []
        }
    },
    {
        name: 'scan_crossplatform',
        description: 'Detect POSIX-only system calls used without platform guards.',
        inputSchema: { type: 'object', properties: {}, required: [] }
    },
    {
        name: 'validate_symbol',
        description: 'Check if a specific API function name is implemented in the source tree.',
        inputSchema: {
            type: 'object',
            properties: {
                symbol: { type: 'string', description: 'Function name to search for, e.g. cublasAxpyEx' }
            },
            required: ['symbol']
        }
    },
    {
        name: 'check_missing',
        description: 'List cuBLAS/cuDNN/cuSPARSE functions declared in known external headers but not implemented.',
        inputSchema: {
            type: 'object',
            properties: {
                api: { type: 'string', description: 'API to check: cublas|cudnn|cusparse|cusolver|cufft|curand|nccl', default: 'cublas' }
            },
            required: []
        }
    }
];

// ─── Tool handlers ────────────────────────────────────────────────────────────

function runBuild(args) {
    const jobs   = args.jobs   || CPU_COUNT;
    const target = args.target || '';
    const cmd    = `cmake --build "${BUILD_DIR}" -j${jobs}` + (target ? ` --target ${target}` : '');
    const result = spawnSync('bash', ['-c', cmd], { cwd: PROJECT_ROOT, encoding: 'utf8', timeout: 300000 });
    const output = (result.stdout || '') + (result.stderr || '');
    const errors   = (output.match(/error:/g)   || []).length;
    const warnings = (output.match(/warning:/g) || []).length;
    const ok       = result.status === 0;
    return {
        success: ok,
        exit_code: result.status,
        errors,
        warnings,
        summary: ok
            ? `Build succeeded. ${warnings} warning(s).`
            : `Build FAILED with ${errors} error(s), ${warnings} warning(s).`,
        output: output.slice(-4000)  // last 4KB
    };
}

function runTests(args) {
    const filter  = args.filter  || '';
    const timeout = args.timeout || 120;
    let cmd = `ctest -j${CPU_COUNT} --output-on-failure --timeout ${timeout}`;
    if (filter) cmd += ` -R "${filter}"`;
    const result = spawnSync('bash', ['-c', cmd], { cwd: BUILD_DIR, encoding: 'utf8', timeout: 600000 });
    const output = (result.stdout || '') + (result.stderr || '');
    const passed = (output.match(/\d+ tests passed/i) || [''])[0];
    const failed = (output.match(/\d+ tests failed/i) || [''])[0];
    return {
        success: result.status === 0,
        passed,
        failed,
        output: output.slice(-6000)
    };
}

function scanStubs(args) {
    const scanPath = path.join(PROJECT_ROOT, args.path || 'src/api');
    const patterns = [
        'TODO', 'FIXME', 'STUB', 'stub', 'not implemented',
        'NOT_IMPLEMENTED', 'placeholder', 'return CUDNN_STATUS_NOT_SUPPORTED',
        'return CUBLAS_STATUS_NOT_SUPPORTED', 'return CUSOLVER_STATUS_NOT_SUPPORTED'
    ];
    const results = {};
    for (const pat of patterns) {
        const res = spawnSync('grep', ['-rn', pat, scanPath, '--include=*.cpp', '--include=*.h'],
            { encoding: 'utf8', timeout: 15000 });
        const lines = (res.stdout || '').split('\n').filter(l => l && !l.includes('//'));
        if (lines.length > 0) results[pat] = lines.slice(0, 20);
    }
    const totalIssues = Object.values(results).reduce((s, v) => s + v.length, 0);
    return {
        total_issues: totalIssues,
        findings: results,
        summary: totalIssues === 0
            ? 'No stubs or TODOs found in ' + (args.path || 'src/api')
            : `Found ${totalIssues} potential issues across ${Object.keys(results).length} pattern(s)`
    };
}

function scanCrossPlatform() {
    // Check for POSIX-only includes without platform guards
    const checks = [
        { desc: 'unguarded <unistd.h>',    cmd: "grep -rn '#include <unistd.h>' src/ --include='*.cpp' | grep -v '#if\\|#ifdef\\|!defined\\|_WIN32'" },
        { desc: 'unguarded <dlfcn.h>',     cmd: "grep -rn '#include <dlfcn.h>' src/ --include='*.cpp' | grep -v '#if\\|#ifdef\\|_WIN32'" },
        { desc: 'unguarded dlopen/dlsym',  cmd: "grep -rn '\\bdlopen\\|\\bdlsym\\b' src/ --include='*.cpp' | grep -v '#ifdef\\|#if defined\\|_WIN32\\|os_backend'" },
        { desc: 'unguarded getpwuid',      cmd: "grep -rn 'getpwuid' src/ --include='*.cpp' | grep -v '#if\\|!defined'" },
        { desc: 'unguarded pthread_setaffinity_np', cmd: "grep -rn 'pthread_setaffinity_np' src/ --include='*.cpp' | grep -v '#if\\|#ifdef'" },
        { desc: 'unguarded immintrin.h',   cmd: "grep -B2 'immintrin.h' src/ -r --include='*.cpp' | grep -v '__AVX\\|__SSE\\|VGRE_HAS\\|x86_64\\|_M_X64\\|ENABLE_VGRE_AMX\\|#if\\|#ifdef'" }
    ];
    const findings = {};
    for (const { desc, cmd } of checks) {
        const res = spawnSync('bash', ['-c', cmd], { cwd: PROJECT_ROOT, encoding: 'utf8', timeout: 10000 });
        const lines = (res.stdout || '').split('\n').filter(Boolean);
        if (lines.length > 0) findings[desc] = lines.slice(0, 10);
    }
    const total = Object.values(findings).reduce((s, v) => s + v.length, 0);
    return {
        total_issues: total,
        findings,
        summary: total === 0
            ? 'No unguarded platform-specific calls found.'
            : `Found ${total} potentially unguarded platform call(s).`
    };
}

function validateSymbol(args) {
    const sym = args.symbol;
    if (!sym) return { found: false, message: 'symbol is required' };
    const res = spawnSync('grep', ['-rn', sym, 'src/', '--include=*.cpp'],
        { cwd: PROJECT_ROOT, encoding: 'utf8', timeout: 10000 });
    const lines = (res.stdout || '').split('\n').filter(Boolean);
    const implLine = lines.find(l => l.includes(`${sym}(`) && !l.includes('//'));
    return {
        found: !!implLine,
        implementation: implLine || null,
        all_references: lines.slice(0, 10),
        summary: implLine
            ? `✓ '${sym}' is implemented at: ${implLine.split(':').slice(0, 2).join(':')}`
            : `✗ '${sym}' NOT FOUND in src/ — likely missing implementation`
    };
}

// Known functions that frameworks commonly expect per API
const EXPECTED_SYMBOLS = {
    cublas: [
        'cublasAxpyEx', 'cublasDotEx', 'cublasDotcEx', 'cublasNrm2Ex',
        'cublasScalEx', 'cublasRotEx',
        'cublasSgemm_v2', 'cublasDgemm_v2', 'cublasHgemm', 'cublasGemmEx',
        'cublasGemmBatchedEx', 'cublasGemmStridedBatchedEx',
        'cublasSgeam', 'cublasDgeam', 'cublasSdgmm', 'cublasDdgmm'
    ],
    cudnn: [
        'cudnnConvolutionForward', 'cudnnConvolutionBackwardData',
        'cudnnConvolutionBackwardFilter', 'cudnnConvolutionBiasActivationForward',
        'cudnnBatchNormalizationForwardTraining', 'cudnnBatchNormalizationBackward',
        'cudnnBatchNormalizationForwardTrainingEx', 'cudnnBatchNormalizationBackwardEx',
        'cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize',
        'cudnnGetBatchNormalizationBackwardExWorkspaceSize',
        'cudnnSoftmaxForward', 'cudnnSoftmaxBackward',
        'cudnnActivationForward', 'cudnnActivationBackward',
        'cudnnRNNForwardTraining', 'cudnnRNNBackwardData', 'cudnnRNNBackwardWeights',
        'cudnnMultiHeadAttnForward', 'cudnnMultiHeadAttnBackwardData',
        'cudnnMultiHeadAttnBackwardWeights', 'cudnnBackendExecute',
        'cudnnCTCLoss', 'cudnnSpatialTfGridGeneratorForward'
    ],
    cusparse: [
        'cusparseSpMM', 'cusparseSpMV', 'cusparseSpGEMM_compute',
        'cusparseSpSM_solve', 'cusparseSpSV_solve', 'cusparseSDDMM',
        'cusparseScsrmv', 'cusparseDcsrmv', 'cusparseAxpby', 'cusparseRot'
    ],
    cusolver: [
        'cusolverDnSgetrf', 'cusolverDnDgetrf', 'cusolverDnSgesvd', 'cusolverDnDgesvd',
        'cusolverDnSpotrf', 'cusolverDnDpotrf', 'cusolverDnSsyevd', 'cusolverDnDsyevd',
        'cusolverDnXgetrf', 'cusolverDnXpotrf', 'cusolverDnXgesvd', 'cusolverDnXsyevd'
    ],
    cufft: ['cufftExecC2C', 'cufftExecR2C', 'cufftExecC2R', 'cufftExecZ2Z', 'cufftExecD2Z', 'cufftExecZ2D'],
    curand: ['curandGenerateUniform', 'curandGenerateNormal', 'curandGeneratePoisson', 'curandCreateGenerator'],
    nccl:  ['ncclAllReduce', 'ncclBroadcast', 'ncclReduce', 'ncclAllGather', 'ncclReduceScatter', 'ncclSend', 'ncclRecv']
};

function checkMissing(args) {
    const api = (args.api || 'cublas').toLowerCase();
    const expected = EXPECTED_SYMBOLS[api];
    if (!expected) return { error: `Unknown API: ${api}. Choose from: ${Object.keys(EXPECTED_SYMBOLS).join(', ')}` };

    const missing = [], found = [];
    for (const sym of expected) {
        const res = spawnSync('grep', ['-rl', sym, 'src/', '--include=*.cpp'],
            { cwd: PROJECT_ROOT, encoding: 'utf8', timeout: 5000 });
        const files = (res.stdout || '').split('\n').filter(Boolean);
        if (files.length > 0) found.push(sym);
        else missing.push(sym);
    }
    return {
        api,
        total_checked: expected.length,
        found_count: found.length,
        missing_count: missing.length,
        missing,
        found,
        summary: missing.length === 0
            ? `All ${expected.length} expected ${api.toUpperCase()} symbols are implemented.`
            : `MISSING ${missing.length}/${expected.length} ${api.toUpperCase()} symbols: ${missing.join(', ')}`
    };
}

// ─── Message handler ─────────────────────────────────────────────────────────

function handleMessage(msg) {
    const { id, method, params } = msg;

    if (method === 'initialize') {
        return sendResult(id, {
            protocolVersion: '2024-11-05',
            capabilities: { tools: {} },
            serverInfo: { name: 'vgre-verify', version: '1.0.0' }
        });
    }

    if (method === 'tools/list') {
        return sendResult(id, { tools: TOOLS });
    }

    if (method === 'tools/call') {
        const name = params.name;
        const args = params.arguments || {};
        let content;
        try {
            let result;
            if      (name === 'build')             result = runBuild(args);
            else if (name === 'test')              result = runTests(args);
            else if (name === 'scan_stubs')        result = scanStubs(args);
            else if (name === 'scan_crossplatform')result = scanCrossPlatform();
            else if (name === 'validate_symbol')   result = validateSymbol(args);
            else if (name === 'check_missing')     result = checkMissing(args);
            else return sendError(id, -32601, `Unknown tool: ${name}`);

            content = [{ type: 'text', text: JSON.stringify(result, null, 2) }];
        } catch (e) {
            content = [{ type: 'text', text: `Error: ${e.message}\n${e.stack}` }];
        }
        return sendResult(id, { content });
    }

    if (method === 'notifications/initialized') return; // no response needed

    if (id !== undefined) sendError(id, -32601, `Method not found: ${method}`);
}

// Startup notification to stderr (not stdout — don't contaminate MCP framing)
process.stderr.write(`[vgre-verify] MCP server started. Project root: ${PROJECT_ROOT}\n`);
