# (c) 2012, Jan-Piet Mens <jpmens(at)gmail.com>
#
# This file is part of Ansible
#
# Ansible is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Ansible is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Ansible.  If not, see <http://www.gnu.org/licenses/>.

from ansible import utils, errors
import os
HAVE_DNS=False
try:
    import dns.resolver
    from dns.exception import DNSException
    HAVE_DNS=True
except ImportError:
    pass

# ==============================================================
# DNSTXT: DNS TXT records
#
#       key=domainname
# TODO: configurable resolver IPs
# --------------------------------------------------------------

class LookupModule(object):

    def __init__(self, basedir=None, **kwargs):
        self.basedir = basedir

        if HAVE_DNS == False:
            raise errors.AnsibleError("Can't LOOKUP(dnstxt): module dns.resolver is not installed")

    def run(self, terms, inject=None, **kwargs):

        terms = utils.listify_lookup_plugin_terms(terms, self.basedir, inject) 

        if isinstance(terms, basestring):
            terms = [ terms ]

        ret = []
        for term in terms:
            domain = term.split()[0]
            string = []
            try:
                answers = dns.resolver.query(domain, 'TXT')
                for rdata in answers:
                    s = rdata.to_text()
                    string.append(s[1:-1])  # Strip outside quotes on TXT rdata

            except dns.resolver.NXDOMAIN:
                string = 'NXDOMAIN'
            except dns.resolver.Timeout:
                string = ''
            except dns.exception.DNSException as  e:
                raise errors.AnsibleError("dns.resolver unhandled exception", e)

            ret.append(''.join(string))
        return ret
