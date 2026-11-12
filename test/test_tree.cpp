#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "xfac_quad/tree/tree.hpp"

#include <set>
#include <stdexcept>
#include <vector>

using xfac_quad::TopologyTree;
using xfac_quad::areEdgesSymmetric;
using xfac_quad::areNeighborsDefined;
using xfac_quad::areNodesInNeigh;
using xfac_quad::isConnected;
using xfac_quad::isRootInNeigh;
using xfac_quad::isTree;
using xfac_quad::isValid;

TEST_CASE("TopologyTree construction", "[tree][construction]")
{
    SECTION("default tree is empty with root zero")
    {
        const TopologyTree tree;

        REQUIRE(tree.root == 0);
        REQUIRE(tree.size() == 0);
        REQUIRE(tree.nodes.empty());
        REQUIRE(tree.neigh.empty());
        REQUIRE_FALSE(isValid(tree));
    }

    SECTION("explicit root does not create nodes")
    {
        const TopologyTree tree(42);

        REQUIRE(tree.root == 42);
        REQUIRE(tree.size() == 0);
        REQUIRE(tree.nodes.empty());
        REQUIRE(tree.neigh.empty());
        REQUIRE_FALSE(isValid(tree));
    }
}

TEST_CASE("TopologyTree edge insertion", "[tree][construction]")
{
    TopologyTree tree(10);

    SECTION("an edge creates both endpoints with symmetric neighbors")
    {
        tree.addEdge(10, 20);

        REQUIRE(tree.size() == 2);
        REQUIRE(tree.neigh.at(10).from_int() == std::vector<int>{20});
        REQUIRE(tree.neigh.at(20).from_int() == std::vector<int>{10});
        REQUIRE(tree.neigh.at(10).pos(20) == 0);
        REQUIRE(tree.neigh.at(20).pos(10) == 0);
        REQUIRE(tree.root == 10);
        REQUIRE(tree.nodes.empty());
    }

    SECTION("branching preserves neighbor insertion order")
    {
        tree.addEdge(10, 30);
        tree.addEdge(10, 20);
        tree.addEdge(10, 40);

        REQUIRE(tree.size() == 4);
        REQUIRE(tree.neigh.at(10).from_int() == std::vector<int>{30, 20, 40});
        REQUIRE(tree.neigh.at(10).pos(30) == 0);
        REQUIRE(tree.neigh.at(10).pos(20) == 1);
        REQUIRE(tree.neigh.at(10).pos(40) == 2);
        for (int leaf : {20, 30, 40}) {
            REQUIRE(tree.neigh.at(leaf).from_int() == std::vector<int>{10});
        }
    }

    SECTION("duplicate edges do not duplicate neighbors")
    {
        tree.addEdge(10, 20);
        tree.addEdge(10, 20);
        tree.addEdge(20, 10);

        REQUIRE(tree.size() == 2);
        REQUIRE(tree.neigh.at(10).from_int() == std::vector<int>{20});
        REQUIRE(tree.neigh.at(20).from_int() == std::vector<int>{10});
    }

    SECTION("self edges are rejected without changing the tree")
    {
        REQUIRE_THROWS_AS(tree.addEdge(10, 10), std::invalid_argument);
        REQUIRE(tree.neigh.empty());

        tree.addEdge(10, 20);
        REQUIRE_THROWS_AS(tree.addEdge(20, 20), std::invalid_argument);
        REQUIRE(tree.size() == 2);
        REQUIRE(tree.neigh.at(10).from_int() == std::vector<int>{20});
        REQUIRE(tree.neigh.at(20).from_int() == std::vector<int>{10});
    }

    REQUIRE(isValid(tree));
}

TEST_CASE("TopologyTree physical nodes are registered separately", "[tree][construction]")
{
    TopologyTree tree(30);
    tree.addEdge(10, 30);
    tree.addEdge(20, 30);
    tree.addEdge(40, 30);
    tree.nodes = {10, 20, 40};

    REQUIRE(tree.size() == 4);
    REQUIRE(tree.root == 30);
    REQUIRE(tree.nodes == std::set<int>{10, 20, 40});
    REQUIRE(tree.nodes.count(30) == 0);
    for (int physical_node : tree.nodes) {
        REQUIRE(tree.neigh.count(physical_node) == 1);
        REQUIRE(tree.neigh.at(physical_node).from_int() == std::vector<int>{30});
    }
    REQUIRE(isValid(tree));
}

TEST_CASE("TopologyTree validation", "[tree][validation]")
{
    TopologyTree tree;
    tree.addEdge(0, 3);
    tree.addEdge(1, 3);
    tree.addEdge(2, 3);
    tree.nodes = {0, 1, 2};

    REQUIRE(areNeighborsDefined(tree));
    REQUIRE(areEdgesSymmetric(tree));
    REQUIRE(isConnected(tree));
    REQUIRE(isTree(tree));
    REQUIRE(areNodesInNeigh(tree));
    REQUIRE(isRootInNeigh(tree));
    REQUIRE(isValid(tree));

    SECTION("undefined neighbors are rejected before traversal")
    {
        tree.neigh.erase(3);

        REQUIRE_FALSE(areNeighborsDefined(tree));
        REQUIRE_FALSE(isValid(tree));
    }

    SECTION("asymmetric edges are rejected")
    {
        tree.neigh.at(3).remove(0);

        REQUIRE_FALSE(areEdgesSymmetric(tree));
        REQUIRE_FALSE(isValid(tree));
    }

    SECTION("disconnected components are rejected")
    {
        tree.addEdge(10, 11);

        REQUIRE_FALSE(isConnected(tree));
        REQUIRE_FALSE(isValid(tree));
    }

    SECTION("cycles are rejected")
    {
        tree.addEdge(0, 1);

        REQUIRE(isConnected(tree));
        REQUIRE_FALSE(isTree(tree));
        REQUIRE_FALSE(isValid(tree));
    }

    SECTION("physical nodes missing from adjacency are rejected")
    {
        tree.nodes.insert(99);

        REQUIRE_FALSE(areNodesInNeigh(tree));
        REQUIRE_FALSE(isValid(tree));
    }

    SECTION("a root missing from adjacency is rejected")
    {
        tree.root = 99;

        REQUIRE_FALSE(isRootInNeigh(tree));
        REQUIRE_FALSE(isValid(tree));
    }
}

TEST_CASE("TopologyTree single-node validation", "[tree][validation]")
{
    TopologyTree tree(42);
    tree.neigh[42];
    tree.nodes.insert(42);

    REQUIRE(isConnected(tree));
    REQUIRE(isTree(tree));
    REQUIRE(isValid(tree));
}

TEST_CASE("Tucker tree construction", "[tree][construction][tucker]")
{
    for (const auto& dimensions : std::vector<std::pair<int, int>>{
             {1, 3}, {2, 1}, {2, 3}, {3, 1}, {4, 2}, {6, 3}}) {
        const auto [dim, nBit] = dimensions;
        CAPTURE(dim, nBit);
        const auto tree = xfac_quad::makeTuckerTree(dim, nBit);
        const auto expected_size = static_cast<std::size_t>(
            dim == 1 ? nBit : dim * (nBit + 1) - 2);

        REQUIRE(tree.root == 0);
        REQUIRE(tree.size() == expected_size);
        REQUIRE(tree.nodes.size() == static_cast<std::size_t>(dim * nBit));
        REQUIRE(isValid(tree));
        for (int node = 0; node < dim * nBit; ++node) {
            REQUIRE(tree.nodes.count(node) == 1);
        }
        for (std::size_t node = tree.nodes.size(); node < tree.size(); ++node) {
            REQUIRE(tree.nodes.count(static_cast<int>(node)) == 0);
            REQUIRE(tree.neigh.at(static_cast<int>(node)).size() == 3);
        }
    }
}

TEST_CASE("Tucker tree rejects nonpositive dimensions", "[tree][construction][tucker]")
{
    REQUIRE_THROWS_AS(xfac_quad::makeTuckerTree(0, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(xfac_quad::makeTuckerTree(-1, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(xfac_quad::makeTuckerTree(2, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(xfac_quad::makeTuckerTree(2, -1), std::invalid_argument);
}
