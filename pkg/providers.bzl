# Copyright 2020 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packaging related providers."""

PackageArtifactInfo = provider(
    doc = """Metadata about a package artifact.""",
    fields = {
        "file_name": "The file name of the artifact.",
        "label": "Label which produced it",
    },
)

PackageVariablesInfo = provider(
    doc = """Variables which may be substituted into package names and content.""",
    fields = {
        "values": "Dict of name/value pairs",
    },
)

# For the below, attributes always look something like this:
#
# ```
#   {"unix": ["0755", "root", "root"]}
# ```
#

PackageFilesInfo = provider(
    doc = """Provider representing the installation of one or more files to destination with attributes""",
    fields = {
        "attributes": """dict of string -> string list: Attributes to apply to installed file(s)""",
        # TODO(nacl): determine what types actually should represent the sources
        # here.  Files, or Labels?

        # This is a mapping of destinations to sources to allow for the same
        # target to be installed to multiple locations within a package within a
        # single provider.
        "dest_src_map": """Map of file destinations to sources""",
    },
)

PackageDirsInfo = provider(
    doc = """Provider representing the creation of one or more directories in a package""",
    fields = {
        "attributes": """dict of string -> string list: Attributes to apply to created directories""",
        "dirs": """string list: installed directory names""",
    },
)

PackageSymlinkInfo = provider(
    doc = """Provider representing the creation of a single symbolic link in a package""",
    fields = {
        "attributes": """dict of string -> string list: Attributes to apply to created symlink""",
        "destination": """string: Filesystem link 'name'""",
        "source": """string or Label: Filesystem link 'target'""",
    },
)

# Grouping provider: the only one that needs to be consumed by packaging (or
# other) rules that materialize paths.
PackageFilegroupInfo = provider(
    doc = """Provider representing a collection of related packaging providers""",
    fields = {
        "pkg_files": "list of child PackageFilesInfo providers",
        "pkg_dirs": "list of child PackageDirInfo providers",
        "pkg_symlinks": "list of child PackageSymlinkInfo providers",
    },
)
