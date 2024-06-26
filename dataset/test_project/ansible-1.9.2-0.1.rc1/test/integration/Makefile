INVENTORY ?= inventory
VARS_FILE ?= integration_config.yml

# Create a semi-random string for use when testing cloud-based resources
ifndef CLOUD_RESOURCE_PREFIX
CLOUD_RESOURCE_PREFIX := $(shell python -c "import string,random; print ('ansible-testing-' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in xrange(8));"))
endif

CREDENTIALS_FILE = credentials.yml
# If credentials.yml exists, use it
ifneq ("$(wildcard $(CREDENTIALS_FILE))","")
CREDENTIALS_ARG = -e @$(CREDENTIALS_FILE)
else
CREDENTIALS_ARG =
endif

# http://unix.stackexchange.com/questions/30091/fix-or-alternative-for-mktemp-in-os-x
MYTMPDIR = $(shell mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')

VAULT_PASSWORD_FILE = vault-password

CONSUL_RUNNING := $(shell python consul_running.py)

all: parsing test_var_precedence unicode test_templating_settings non_destructive destructive includes check_mode test_hash test_handlers test_group_by test_vault test_tags

parsing:
	ansible-playbook bad_parsing.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -vvv $(TEST_FLAGS) --tags prepare,common,scenario1; [ $$? -eq 3 ]
	ansible-playbook bad_parsing.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -vvv $(TEST_FLAGS) --tags prepare,common,scenario2; [ $$? -eq 3 ]
	ansible-playbook bad_parsing.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -vvv $(TEST_FLAGS) --tags prepare,common,scenario3; [ $$? -eq 3 ]
	ansible-playbook bad_parsing.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -vvv $(TEST_FLAGS) --tags prepare,common,scenario4; [ $$? -eq 3 ]
	ansible-playbook bad_parsing.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -vvv $(TEST_FLAGS) --tags prepare,common,scenario5; [ $$? -eq 3 ]
	ansible-playbook good_parsing.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS)

includes:
	ansible-playbook test_includes.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) $(TEST_FLAGS)

unicode:
	ansible-playbook unicode.yml -i $(INVENTORY) -e @$(VARS_FILE) -v $(TEST_FLAGS) -e 'extra_var=café'
	# Test the start-at-task flag #9571
	ansible-playbook unicode.yml -i $(INVENTORY) -e @$(VARS_FILE) -v --start-at-task '*¶' -e 'start_at_task=True' $(TEST_FLAGS)

test_templating_settings:
	ansible-playbook test_templating_settings.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS)

non_destructive:
	ansible-playbook non_destructive.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS)

destructive:
	ansible-playbook destructive.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS)

check_mode:
	ansible-playbook check_mode.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v --check $(TEST_FLAGS)

test_group_by:
	ansible-playbook test_group_by.yml -i inventory.group_by -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS)

test_handlers:
	ansible-playbook test_handlers.yml -i inventory.handlers -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS)
	# Not forcing, should only run on successful host
	[ "$$(ansible-playbook test_force_handlers.yml --tags normal -i inventory.handlers -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) | egrep -o CALLED_HANDLER_. | sort | uniq | xargs)" = "CALLED_HANDLER_B" ]
	# Forcing from command line
	[ "$$(ansible-playbook test_force_handlers.yml --tags normal -i inventory.handlers --force-handlers -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) | egrep -o CALLED_HANDLER_. | sort | uniq | xargs)" = "CALLED_HANDLER_A CALLED_HANDLER_B" ]
	# Forcing from command line, should only run later tasks on unfailed hosts
	[ "$$(ansible-playbook test_force_handlers.yml --tags normal -i inventory.handlers --force-handlers -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) | egrep -o CALLED_TASK_. | sort | uniq | xargs)" = "CALLED_TASK_B CALLED_TASK_D CALLED_TASK_E" ]
	# Forcing from command line, should call handlers even if all hosts fail
	[ "$$(ansible-playbook test_force_handlers.yml --tags normal -i inventory.handlers --force-handlers -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v -e fail_all=yes $(TEST_FLAGS) | egrep -o CALLED_HANDLER_. | sort | uniq | xargs)" = "CALLED_HANDLER_A CALLED_HANDLER_B" ]
	# Forcing from ansible.cfg
	[ "$$(ANSIBLE_FORCE_HANDLERS=true ansible-playbook --tags normal test_force_handlers.yml -i inventory.handlers -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) | egrep -o CALLED_HANDLER_. | sort | uniq | xargs)" = "CALLED_HANDLER_A CALLED_HANDLER_B" ]
	# Forcing true in play
	[ "$$(ansible-playbook test_force_handlers.yml --tags force_true_in_play -i inventory.handlers -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) | egrep -o CALLED_HANDLER_. | sort | uniq | xargs)" = "CALLED_HANDLER_A CALLED_HANDLER_B" ]
	# Forcing false in play, which overrides command line
	[ "$$(ansible-playbook test_force_handlers.yml --force-handlers --tags force_false_in_play -i inventory.handlers -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) | egrep -o CALLED_HANDLER_. | sort | uniq | xargs)" = "CALLED_HANDLER_B" ]

test_hash:
	ANSIBLE_HASH_BEHAVIOUR=replace ansible-playbook test_hash.yml -i $(INVENTORY) $(CREDENTIALS_ARG) -v -e '{"test_hash":{"extra_args":"this is an extra arg"}}'
	ANSIBLE_HASH_BEHAVIOUR=merge ansible-playbook test_hash.yml -i $(INVENTORY) $(CREDENTIALS_ARG) -v -e '{"test_hash":{"extra_args":"this is an extra arg"}}'

test_var_precedence:
	ansible-playbook test_var_precedence.yml -i $(INVENTORY) $(CREDENTIALS_ARG) -v -e 'extra_var=extra_var' -e 'extra_var_override=extra_var_override'

test_vault:
	ansible-playbook test_vault.yml -i $(INVENTORY) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) --vault-password-file $(VAULT_PASSWORD_FILE) --list-tasks
	ansible-playbook test_vault.yml -i $(INVENTORY) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) --vault-password-file $(VAULT_PASSWORD_FILE) --list-hosts
	ansible-playbook test_vault.yml -i $(INVENTORY) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) --vault-password-file $(VAULT_PASSWORD_FILE) --syntax-check
	ansible-playbook test_vault.yml -i $(INVENTORY) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) --vault-password-file $(VAULT_PASSWORD_FILE)

# test_delegate_to does not work unless we have permission to ssh to localhost.
# Would take some more effort on our test systems to implement that -- probably
# the test node should create an ssh public-private key pair that allows the
# root user on a node to ssh to itself.  Until then, this is not in make all.
# Have to run it manually.  Ordinary users should be able to run this test as
# long as they have permissions to login to their local machine via ssh.
test_delegate_to:
	ansible-playbook test_delegate_to.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS)

test_winrm:
	ansible-playbook test_winrm.yml -i inventory.winrm -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS)

test_tags:
	# Run everything by default
	[ "$$(ansible-playbook --list-tasks test_tags.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) | fgrep Task_with | xargs)" = "Task_with_tag TAGS: [tag] Task_with_always_tag TAGS: [always] Task_without_tag TAGS: []" ]
	# Run the exact tags, and always
	[ "$$(ansible-playbook --list-tasks --tags tag test_tags.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) | fgrep Task_with | xargs)" = "Task_with_tag TAGS: [tag] Task_with_always_tag TAGS: [always]" ]
	# Skip one tag
	[ "$$(ansible-playbook --list-tasks --skip-tags tag test_tags.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -v $(TEST_FLAGS) | fgrep Task_with | xargs)" = "Task_with_always_tag TAGS: [always] Task_without_tag TAGS: []" ]


cloud: amazon rackspace

cloud_cleanup: amazon_cleanup rackspace_cleanup

amazon_cleanup:
	python cleanup_ec2.py -y --match="^$(CLOUD_RESOURCE_PREFIX)"

gce_setup:
	python setup_gce.py "$(CLOUD_RESOURCE_PREFIX)"

gce_cleanup:
	python cleanup_gce.py -y --match="^$(CLOUD_RESOURCE_PREFIX)"

rackspace_cleanup:
	python cleanup_rax.py -y --match="^$(CLOUD_RESOURCE_PREFIX)"

$(CREDENTIALS_FILE):
	@echo "No credentials file found.  A file named '$(CREDENTIALS_FILE)' is needed to provide credentials needed to run cloud tests.  See sample 'credentials.template' file."
	@exit 1

amazon: $(CREDENTIALS_FILE)
	ANSIBLE_HOST_KEY_CHECKING=False ANSIBLE_SSH_PIPELINING=no BOTO_CONFIG=/dev/null ansible-playbook amazon.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -e "resource_prefix=$(CLOUD_RESOURCE_PREFIX)" -v $(TEST_FLAGS) ; \
    RC=$$? ; \
    CLOUD_RESOURCE_PREFIX="$(CLOUD_RESOURCE_PREFIX)" make amazon_cleanup ; \
    exit $$RC;

gce: $(CREDENTIALS_FILE)
	CLOUD_RESOURCE_PREFIX="$(CLOUD_RESOURCE_PREFIX)" make gce_setup ; \
    ansible-playbook gce.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -e "resource_prefix=$(CLOUD_RESOURCE_PREFIX)" -v $(TEST_FLAGS) ; \
    RC=$$? ; \
    CLOUD_RESOURCE_PREFIX="$(CLOUD_RESOURCE_PREFIX)" make gce_cleanup ; \
    exit $$RC;

rackspace: $(CREDENTIALS_FILE)
	ansible-playbook rackspace.yml -i $(INVENTORY) -e @$(VARS_FILE) $(CREDENTIALS_ARG) -e "resource_prefix=$(CLOUD_RESOURCE_PREFIX)" -v $(TEST_FLAGS) ; \
    RC=$$? ; \
    CLOUD_RESOURCE_PREFIX="$(CLOUD_RESOURCE_PREFIX)" make rackspace_cleanup ; \
    exit $$RC;

$(CONSUL_RUNNING):

consul:
ifeq ($(CONSUL_RUNNING), True)
	ansible-playbook -i $(INVENTORY) consul.yml ; \
	ansible-playbook -i ../../plugins/inventory/consul_io.py consul_inventory.yml
else
	@echo "Consul agent is not running locally. To run a cluster locally see http://github.com/sgargan/consul-vagrant"
endif

test_galaxy: test_galaxy_spec test_galaxy_yaml

test_galaxy_spec:
	mytmpdir=$(MYTMPDIR) ; \
	ansible-galaxy install -r galaxy_rolesfile -p $$mytmpdir/roles ; \
    cp galaxy_playbook.yml $$mytmpdir ; \
    ansible-playbook -i $(INVENTORY) $$mytmpdir/galaxy_playbook.yml -v $(TEST_FLAGS) ; \
    RC=$$? ; \
    rm -rf $$mytmpdir ; \
    exit $$RC

test_galaxy_yaml:
	mytmpdir=$(MYTMPDIR) ; \
	ansible-galaxy install -r galaxy_roles.yml -p $$mytmpdir/roles ; \
    cp galaxy_playbook.yml $$mytmpdir ; \
    ansible-playbook -i $(INVENTORY) $$mytmpdir/galaxy_playbook.yml -v $(TEST_FLAGS) ; \
    RC=$$? ; \
    rm -rf $$mytmpdir ; \
    exit $$RC
