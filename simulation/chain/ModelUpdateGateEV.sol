// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ModelUpdateGateEV — e-value verification of self-updating IDS models
/// @notice Gateways commit drift-triggered model updates (hashes only; weights stay off-chain).
///         Commit before randomness: the audit seed keccak256(modelHash, blockhash(commitBlock + 1))
///         exists only after the block following the commitment, so a gateway cannot tune its
///         candidate to the flows that will be audited; validators may attest only once it exists.
///         Each validator tests every cell (class x private sub-cluster) of its clean data with an
///         anytime-valid e-process of H0 "candidate is eps-worse than the reference" and votes
///           Pass   all cells crossed the e-value threshold (n - f) * (pi^2 / 6) * k^2 / alpha
///                  for the gateway's k-th update (alpha spending: lifetime error <= alpha);
///                  the contract checks the reported smallest e-value against this threshold
///           Harm   some cell's estimated loss exceeds its tolerance
///           Unsure evidence insufficient
///         f + 1 passes accept (at least one honest validator passed, so with at most f Byzantine
///         validators P(accept | eps-harmful) <= alpha_k), f + 1 harms reject, otherwise the update
///         is deferred (the gateway keeps training the shadow candidate and commits it again).
///         One open update per gateway bounds proposal flooding.
///         Coverage obligation: when f + 1 validators report that a class they hold is not
///         detected by the serving model (heartbeat audits), CoverageEscalation is emitted.
contract ModelUpdateGateEV {
    enum Status { None, Pending, Accepted, Rejected, Deferred }
    enum Verdict { Pass, Unsure, Harm }

    struct Update {
        address gateway;
        bytes32 modelHash;
        bytes32 parentHash;
        uint64 commitBlock;
        uint32 k;
        uint8 passes;
        uint8 fails;
        uint8 harms;
        Status status;
    }

    uint256 private constant PI2_OVER_6_PPM = 1644934;  // pi^2 / 6 in parts per million

    address public owner;
    uint8 public n;
    uint8 public f;
    uint32 public alphaPpm;

    mapping(address => bool) public isValidator;
    mapping(address => bool) public isGateway;
    mapping(address => bytes32) public activeModel;
    mapping(address => uint32) public proposals;          // k: per-gateway update counter
    mapping(address => uint256) public openUpdate;        // 1 + id of the pending update, 0 = none
    mapping(uint256 => mapping(address => bool)) public attested;
    mapping(bytes32 => mapping(address => bool)) private coverageVoted;
    mapping(bytes32 => uint8) public coverageVotes;
    Update[] public updates;

    event GatewayRegistered(address indexed gateway, bytes32 genesisModel);
    event UpdateProposed(uint256 indexed id, address indexed gateway, bytes32 modelHash,
                         bytes32 parentHash, uint64 commitBlock, uint32 k);
    event UpdateAttested(uint256 indexed id, address indexed validator, Verdict verdict, uint256 eMinMilli);
    event UpdateAccepted(uint256 indexed id, address indexed gateway, bytes32 modelHash);
    event UpdateRejected(uint256 indexed id, address indexed gateway, bytes32 modelHash);
    event UpdateDeferred(uint256 indexed id, address indexed gateway, bytes32 modelHash);
    event CoverageEscalation(address indexed gateway, uint16 classId, uint64 epoch);

    modifier onlyOwner() { require(msg.sender == owner, "not owner"); _; }

    constructor(uint8 _n, uint8 _f, uint32 _alphaPpm) {
        require(_n > 2 * _f, "need n > 2f");
        owner = msg.sender;
        n = _n;
        f = _f;
        alphaPpm = _alphaPpm;
    }

    function addValidator(address v) external onlyOwner { isValidator[v] = true; }

    function registerGateway(address g, bytes32 genesisModel) external onlyOwner {
        isGateway[g] = true;
        activeModel[g] = genesisModel;
        emit GatewayRegistered(g, genesisModel);
    }

    /// @notice e-value threshold for the k-th update, x1000 (milli)
    function eThresholdMilli(uint32 k) public view returns (uint256) {
        return uint256(n - f) * PI2_OVER_6_PPM * uint256(k) * uint256(k) * 1000 / alphaPpm;
    }

    function proposeUpdate(bytes32 modelHash) external returns (uint256 id) {
        require(isGateway[msg.sender], "not gateway");
        uint256 open = openUpdate[msg.sender];
        require(open == 0 || updates[open - 1].status != Status.Pending, "update already open");
        uint32 k = ++proposals[msg.sender];
        id = updates.length;
        updates.push(Update(msg.sender, modelHash, activeModel[msg.sender], uint64(block.number), k,
                            0, 0, 0, Status.Pending));
        openUpdate[msg.sender] = id + 1;
        emit UpdateProposed(id, msg.sender, modelHash, activeModel[msg.sender], uint64(block.number), k);
    }

    /// @notice audit randomness, available once the block after the commitment is mined
    function auditSeed(uint256 id) public view returns (bytes32) {
        Update storage u = updates[id];
        require(block.number > u.commitBlock + 1, "seed not revealed yet");
        return keccak256(abi.encodePacked(u.modelHash, blockhash(u.commitBlock + 1)));
    }

    function attest(uint256 id, Verdict verdict, uint256 eMinMilli) external {
        require(isValidator[msg.sender], "not validator");
        Update storage u = updates[id];
        require(u.status == Status.Pending, "not pending");
        require(block.number > u.commitBlock + 1, "audit seed not revealed yet");
        require(!attested[id][msg.sender], "already attested");
        if (verdict == Verdict.Pass) {
            require(eMinMilli >= eThresholdMilli(u.k), "e-value below threshold");
        }
        attested[id][msg.sender] = true;
        emit UpdateAttested(id, msg.sender, verdict, eMinMilli);

        if (verdict == Verdict.Pass) {
            u.passes += 1;
        } else {
            u.fails += 1;
            if (verdict == Verdict.Harm) u.harms += 1;
        }
        if (u.passes >= f + 1) {
            u.status = Status.Accepted;
            activeModel[u.gateway] = u.modelHash;
            emit UpdateAccepted(id, u.gateway, u.modelHash);
        } else if (u.harms >= f + 1) {
            u.status = Status.Rejected;
            emit UpdateRejected(id, u.gateway, u.modelHash);
        } else if (u.fails >= n - f) {
            u.status = Status.Deferred;
            emit UpdateDeferred(id, u.gateway, u.modelHash);
        }
    }

    function reportCoverage(address gateway, uint16 classId, uint64 epoch) external {
        require(isValidator[msg.sender], "not validator");
        bytes32 key = keccak256(abi.encodePacked(gateway, classId, epoch));
        require(!coverageVoted[key][msg.sender], "already reported");
        coverageVoted[key][msg.sender] = true;
        if (++coverageVotes[key] == f + 1) {
            emit CoverageEscalation(gateway, classId, epoch);
        }
    }

    function updateCount() external view returns (uint256) { return updates.length; }

    function status(uint256 id) external view returns (Status) { return updates[id].status; }
}
