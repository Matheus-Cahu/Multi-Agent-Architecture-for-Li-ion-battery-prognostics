// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract ReportAnchor {
    mapping(bytes32 => bytes32) private hashes;   // reportId => contentHash

    event Anchored(bytes32 indexed reportId, bytes32 contentHash);

    function anchor(bytes32 reportId, bytes32 contentHash) external {
        require(hashes[reportId] == bytes32(0), "ja ancorado");
        hashes[reportId] = contentHash;
        emit Anchored(reportId, contentHash);
    }

    function getHash(bytes32 reportId) external view returns (bytes32) {
        return hashes[reportId];
    }
}
