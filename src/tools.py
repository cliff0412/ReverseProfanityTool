import copy
import os.path
import traceback

import pyopencl as cl
import numpy as np
from src.precomp import g_precomp
import ecdsa
import struct
import ctypes
import pandas as pd
from sqlalchemy import create_engine, Column, String, Integer, Boolean, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import time
import web3
import hashlib
import requests
from eth_account._utils.legacy_transactions import ALLOWED_TRANSACTION_KEYS
from eth_account._utils.typed_transactions import TYPED_TRANSACTION_FORMATTERS
from eth_account._utils.signing import serializable_unsigned_transaction_from_dict
from eth_account._utils.signing import extract_chain_id, to_standard_v
import json
import multiprocessing
import struct
import random


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

uint64 = np.dtype([("x", "<u4"), ("y", "<u4")])

def binary_search(target1, target2, mm):
    low = 0
    high = len(mm) - 1
    step = 0

    while (low <= high):
        mid = (low + high) // 2
        midVal = mm[mid]
        step = step + 1
        if ((midVal[2] > target1) or (midVal[2] == target1 and midVal[1] > target2)):
            high = mid - 1
        elif ((midVal[2] < target1) or (midVal[2] == target1 and midVal[1] < target2)):
            low = mid + 1
        else:
            return midVal
    return False

def private_k_recover(seed: int, batch: int, global_id: int) -> bytes:
    m = mt19937_64()
    m.seed(seed)
    k = bytearray(32)
    struct.pack_into('>Q', k, 24, m.int64() + batch + 2)
    struct.pack_into('>Q', k, 16, m.int64())
    struct.pack_into('>Q', k, 8, m.int64())
    struct.pack_into('>Q', k, 0, m.int64() + global_id)
    return bytes(k)

class mt19937_64(object):
    def __init__(self):
        self.mt = [0]*312
        self.mti = 313

    def seed(self, seed):
        self.mt[0] = seed & 0xffffffffffffffff
        for i in range(1,312):
            self.mt[i] = (6364136223846793005 * (self.mt[i-1] ^ (self.mt[i-1] >> 62)) + i) & 0xffffffffffffffff
        self.mti = 312

    def init_by_array(self, key):
        self.seed(19650218)
        i = 1
        j = 0
        k = max(312, len(key))
        for ki in range(k):
            self.mt[i] = ((self.mt[i] ^ ((self.mt[i-1] ^ (self.mt[i-1] >> 62)) * 3935559000370003845)) + key[j] + j) & 0xffffffffffffffff
            i += 1
            j += 1
            if i >= 312:
                self.mt[0] = self.mt[311]
                i = 1
            if j >= len(key):
                j = 0
        for ki in range(312):
            self.mt[i] = ((self.mt[i] ^ ((self.mt[i-1] ^ (self.mt[i-1] >> 62)) * 2862933555777941757)) - i) & 0xffffffffffffffff
            i += 1
            if i >= 312:
                self.mt[0] = self.mt[311]
                i = 1
        self.mt[0] = 1 << 63

    def int64(self):
        if self.mti >= 312:
            if self.mti == 313:
                self.seed(5489)

            for k in range(311):
                y = (self.mt[k] & 0xFFFFFFFF80000000) | (self.mt[k+1] & 0x7fffffff)
                if k < 312 - 156:
                    self.mt[k] = self.mt[k+156] ^ (y >> 1) ^ (0xB5026F5AA96619E9 if y & 1 else 0)
                else:
                    self.mt[k] = self.mt[k+156-624] ^ (y >> 1) ^ (0xB5026F5AA96619E9 if y & 1 else 0)

            y = (self.mt[311] & 0xFFFFFFFF80000000) | (self.mt[0] & 0x7fffffff)
            self.mt[311] = self.mt[155] ^ (y >> 1) ^ (0xB5026F5AA96619E9 if y & 1 else 0)
            self.mti = 0

        y = self.mt[self.mti]
        self.mti += 1

        y ^= (y >> 29) & 0x5555555555555555
        y ^= (y << 17) & 0x71D67FFFEDA60000
        y ^= (y << 37) & 0xFFF7EEE000000000
        y ^= (y >> 43)

        return y

    def int64b(self):
        if self.mti == 313:
            self.seed(5489)

        k = self.mti

        if k == 312:
            k = 0
            self.mti = 0

        if k == 311:
            y = (self.mt[311] & 0xFFFFFFFF80000000) | (self.mt[0] & 0x7fffffff)
            self.mt[311] = self.mt[155] ^ (y >> 1) ^ (0xB5026F5AA96619E9 if y & 1 else 0)
        else:
            y = (self.mt[k] & 0xFFFFFFFF80000000) | (self.mt[k+1] & 0x7fffffff)
            if k < 312 - 156:
                self.mt[k] = self.mt[k+156] ^ (y >> 1) ^ (0xB5026F5AA96619E9 if y & 1 else 0)
            else:
                self.mt[k] = self.mt[k+156-624] ^ (y >> 1) ^ (0xB5026F5AA96619E9 if y & 1 else 0)

        y = self.mt[self.mti]
        self.mti += 1

        y ^= (y >> 29) & 0x5555555555555555
        y ^= (y << 17) & 0x71D67FFFEDA60000
        y ^= (y << 37) & 0xFFF7EEE000000000
        y ^= (y >> 43)

        return y


class RevProfanity:
    Base = declarative_base()

    class Collision(Base):
        __tablename__ = "collisions"
        __table_args__ = {'extend_existing': True}
        pk_hash = Column(BigInteger, primary_key=True)
        address = Column(String)
        public_key = Column(String)
        start_procentage = Column(Integer)
        end_procentage = Column(Integer)
        id_run = Column(Integer)
        iteration = Column(Integer)
        target_1 = Column(BigInteger)
        target_2 = Column(BigInteger)

    def __init__(self,
                 blockchain: str = 'ethereum'):
        self.engine = create_engine("sqlite:///DB/Profanity.db")
        self.Session = sessionmaker(bind=self.engine)
        self.Base.metadata.create_all(self.engine, checkfirst=True)

        self.num_points = 16384 * 255
        self.global_size = (self.num_points,)
        # self.global_size = (1,)
        self.local_size = None
        self.iterations = 2000

        # Create a context and command queue
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.point_dtype = None

        self.elems = None
        self.start_percent = None
        self.end_percent = None
        client_data = json.load(open("config.json"))
        self.w3 = web3.Web3(web3.HTTPProvider(client_data[blockchain]['w3']))
        self.etherscan_api_key = client_data[blockchain]['etherscan_key']
        self.base_url = client_data[blockchain]['base_url']

    @staticmethod
    def sub_key(pub_key):

        bytes_representation = b'\x04' + bytes.fromhex(
            "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798") + bytes.fromhex(
            "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8")

        # Create a new secp256k1 public key
        g = ecdsa.VerifyingKey.from_string(bytes_representation, curve=ecdsa.SECP256k1)

        # Generate 16384 * 255 starting public keys
        m = 16384 * 255 + - 1
        scalar = 1 << 192
        pk = (scalar * m) * g.pubkey.point

        pub_key = ecdsa.VerifyingKey.from_string(
            b'\x04' + bytes.fromhex(pub_key[:64]) + bytes.fromhex(pub_key[64:]), curve=ecdsa.SECP256k1)
        pub_key_sub = ecdsa.VerifyingKey.from_string(
            b'\x04' + bytes.fromhex(str(hex(pk.x()))[2:]) + bytes.fromhex(str(hex(pk.y()))[2:]), curve=ecdsa.SECP256k1)

        pk_neg = pub_key.pubkey.point + pub_key_sub.pubkey.point.__neg__()
        return pk_neg.to_bytes().hex()

    @staticmethod
    def point_addition(pub_key, iteration, id_run):
        pub_key = copy.copy(pub_key)
        bytes_representation = b'\x04' + bytes.fromhex(
            "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798") + bytes.fromhex(
            "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8")

        # Create a new secp256k1 public key
        g = ecdsa.VerifyingKey.from_string(bytes_representation, curve=ecdsa.SECP256k1)

        scalar = 1 << 192
        add_pk = (scalar * id_run - iteration) * g.pubkey.point
        # print(hex(add_pk.x())[2:], str(hex(add_pk.y()))[2:])
        pub_key = ecdsa.VerifyingKey.from_string(
            b'\x04' + bytes.fromhex(pub_key[:64]) + bytes.fromhex(pub_key[64:]), curve=ecdsa.SECP256k1)
        pub_key_add = ecdsa.VerifyingKey.from_string(
            b'\x04' + bytes.fromhex('0'*(66-len(hex(add_pk.x())))+hex(add_pk.x())[2:]) + bytes.fromhex('0'*(66-len(hex(add_pk.y())))+hex(add_pk.y())[2:]), curve=ecdsa.SECP256k1)

        pk_added = pub_key.pubkey.point + pub_key_add.pubkey.point
        return pk_added

    def check_collisions(self, pub_key):
        print('Checking collisions for public key: ', pub_key)
        # Select all rows containing public key
        data = pd.read_sql("""select * from collisions where public_key = '{}' and (iteration+id_run+target_1+target_2) > 0""".format(pub_key), self.engine)
        # Get the starting pub key for backtracking profanity
        file_size = os.path.getsize('output.bin') // 12
        mm = np.memmap('output.bin', dtype="<u4", mode='r', shape=(file_size, 3))

        # For each row we check if calculations are correct and last 8bytes of X coordinate match
        for ind, row in data.iterrows():
            pub_key_start = self.point_addition(pub_key, row.iteration, row.id_run+1)
            if hex(row.target_1)[2:] in hex(pub_key_start.x()) and hex(row.target_2)[2:] in hex(pub_key_start.x()):
                # Run binary search
                seed = binary_search(row.target_1, row.target_2, mm)[0]
                priv_key = private_k_recover(int(seed),
                                             row.iteration - 2,
                                             self.num_points - row.id_run - 2)
                address = web3.Account.from_key(priv_key).address
                # print(address, row.address)
                if address.lower() == row.address.lower():
                    print('Address: ', row.address)
                    print('Recovered key: ', priv_key.hex())





    # Function to convert a chunk of data to the custom data type
    @staticmethod
    def convert_chunk(chunk):

        # View the chunk as the new type
        converted = chunk.view(uint64)

        # Return the converted chunk
        return converted

    @staticmethod
    def get_pub_from_priv(private_key):
        import ecdsa
        # Create an ECDSA signing object using the SECP256k1 curve
        signing_key = ecdsa.SigningKey.from_string(bytes.fromhex(private_key), curve=ecdsa.SECP256k1)
        # Get the public key from the signing object
        public_key = signing_key.get_verifying_key().to_string()
        # Encode the public key as a hexadecimal string
        public_key_hex = public_key.hex()
        return public_key_hex

    @timeit
    def _read_8byte_pubkey(self,
                           start_percent,
                           end_percent,
                           file_name,
                           verbosity=True):
        # check if percent is within valid range
        if start_percent < 0 or start_percent > 100:
            raise ValueError("percent must be between 0 and 100")
        if end_percent < 0 or end_percent > 100:
            raise ValueError("percent must be between 0 and 100")

        file_size = os.path.getsize(file_name) // 12
        start_num = int(file_size * start_percent / 100)
        end_num = int(file_size * end_percent / 100)

        mm = np.memmap(file_name, dtype="<u4", mode='r', shape=(file_size, 3))
        # Read some portion of the file
        data = mm[start_num:end_num, 2:0:-1]
        data_contiguous = np.ascontiguousarray(data)

        return data_contiguous.view(uint64), start_percent, end_percent

    def load_sorted_keys(self, st, en, str_nam):
        self.elems, self.start_percent, self.end_percent = self._read_8byte_pubkey(start_percent=st,
                                                                                   end_percent=en,
                                                                                   file_name=str_nam,
                                                                                   verbosity=True)

    def _get_pubkey(self, tx, get_tx=True, verbose=False):
        if get_tx:
            tx = self.w3.eth.get_transaction(tx)

        s = self.w3.eth.account._keys.Signature(vrs=(
            to_standard_v(extract_chain_id(tx.v)[1]),
            self.w3.to_int(tx.r),
            self.w3.to_int(tx.s)
        ))

        if 'maxFeePerGas' not in dict(tx).keys():

            if 'to' not in dict(tx).keys():
                _keys = ALLOWED_TRANSACTION_KEYS - {'chainId', 'data', 'to'}
            else:
                _keys = ALLOWED_TRANSACTION_KEYS - {'chainId', 'data'}
            tt = {k: tx[k] for k in _keys}
            tt['data'] = tx.input
            tt['chainId'] = extract_chain_id(tx.v)[0]

            ut = serializable_unsigned_transaction_from_dict(tt)
            if verbose:
                print("signature: ", s)
                print("Transaction: ", tt)
                print("Hash:: ", ut.hash())
                print("Public kye: ", str(s.recover_public_key_from_msg_hash(ut.hash())))
        else:
            if 'to' not in dict(tx).keys():
                _keys = set(TYPED_TRANSACTION_FORMATTERS.keys()) - {'chainId', 'data', 'gasPrice', 'accessList', 'to'}
            else:
                _keys = set(TYPED_TRANSACTION_FORMATTERS.keys()) - {'chainId', 'data', 'gasPrice', 'accessList'}
            tt = {k: TYPED_TRANSACTION_FORMATTERS[k](tx[k]) for k in _keys}
            tt['data'] = tx.input
            tt['chainId'] = tx.chainId
            ut = serializable_unsigned_transaction_from_dict(tt)
            if verbose:
                print("signature: ", s)
                print("Transaction: ", tt)
                print("Hash:: ", ut.hash())
                print("Public key: ", str(s.recover_public_key_from_msg_hash(ut.hash())))

        pub_key = str(s.recover_public_key_from_msg_hash(ut.hash()))
        hash = self.w3.keccak(hexstr=pub_key)
        address = self.w3.to_hex(hash[-20:])
        return pub_key, hash, address

    def get_pubkeys(self,
                    eoas: list,
                    txs: list):
        addresses = []
        public_keys = []
        for ind, tx in enumerate(txs):
            try:
                res = self._get_pubkey(tx)
                assert res[2].lower() == eoas[ind].lower()
                public_keys.append(res[0])
                addresses.append(eoas[ind])
            except Exception as e:
                traceback.print_exc()
                print(e)
                print(eoas[ind])
        addresses = pd.DataFrame(columns=['address', 'public_key'], data=zip(addresses, public_keys))
        return addresses

    def get_tx_etherscan(self, addresses: list):
        params = {
            'module': 'account',
            'action': 'txlist',
            'startblock': '0',
            'endblock': '99999999',
            'sort': 'desc',
            'apikey': self.etherscan_api_key
        }

        txs = []
        # Loop through each address
        cnt = 0
        l_a = len(addresses)
        addresses_no_contracts = []
        for ind, address in enumerate(addresses):
            # Make the API call to get the latest transactions
            params['address'] = address
            response = requests.get(
                self.base_url,
                params=params
            )

            if response.status_code == 200:
                transactions = response.json()['result']
                transactions = list(filter(lambda x: address.lower() == x['from'], transactions))
            else:
                raise Exception('Failed to fetch transaction from etherscan: ', response.status_code)

            if transactions:
                if (address.lower() == transactions[0]['to'].lower() and transactions[0]['input'] == '0x') or \
                        address.lower() == transactions[0]['from'].lower():
                    tx = transactions[0]
                    txs.append(tx["hash"])
                    addresses_no_contracts.append(self.w3.to_checksum_address(address))
                else:
                    print('Contract: ', address)
            time.sleep(.2)
            if ind / l_a > cnt * .1:
                print(ind / len(addresses) * 100, " %")
                cnt += 1
        print('Count of addresses: ', len(txs))
        return addresses_no_contracts, txs

    def _get_etherscan_deployers(self, contract_addresses: list):
        """

        :param contract_addresses: Pass a list of contract addresses to get EOAs from etherscan
        :return: A list of
        """
        params = {"module": "contract", "action": "getcontractcreation",
                  "contractaddresses": ",".join(contract_addresses), "apikey": self.etherscan_api_key}
        response = requests.get(self.base_url, params=params)

        if response.status_code == 200:
            result = response.json()["result"]
            r = []
            for res in result:
                r.append(res['contractCreator'])
        else:
            raise Exception("Failed to fetch deployer address")
        return r

    def get_deployers(self, contract_addresses: list):
        """
        Iterate through contract_addresses in chunks of 5 (etherscan limit)
        :param contract_addresses: list of contract addresses to retrieve deployer EOAs
        :return:
        """
        list_len = len(contract_addresses)
        counter = 0
        eoas = []
        print('Fetching deployers, contract count: ', list_len)
        while counter < list_len:
            res = self._get_etherscan_deployers(contract_addresses[counter:min(counter + 5, list_len)])
            eoas.extend(res)
            time.sleep(.2)
            counter += 5
        eoas = list(set(eoas))
        return eoas

    @timeit
    def build_kernel(self,
                     kernel_file='src/opencl_code.cl'):
        kernel = ''.join(open(kernel_file, 'r').readlines())

        # Define the point struct
        self.point_dtype = np.dtype([("x", np.uint32, 8), ("y", np.uint32, 8)], align=True)

        # Create an empty array of points
        self.precomp = np.empty(len(g_precomp), dtype=self.point_dtype)
        # Convert the nested list to an array of points
        for i, (x, y) in enumerate(g_precomp):
            self.precomp[i]["x"] = np.array(x, dtype=np.uint32)
            self.precomp[i]["y"] = np.array(y, dtype=np.uint32)

        # Create a buffer to hold the data on the device
        print('Creating buffer elems')
        # self.set_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, ctypes.sizeof(self.elems))
        self.set_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.elems)
        print('Creating buffer precomp')
        self.precomp_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, self.precomp.nbytes)

        # Copy the data from the host to the device
        print('Copying to device elems')
        cl.enqueue_copy(self.queue, self.set_buf, self.elems)
        # kernel.set_arg(0, self.set_buf)
        # cl.enqueue_nd_range_kernel(self.queue, kernel, (self.global_size, ), (self.local_size, ))
        print('Copying to device precomp')
        cl.enqueue_copy(self.queue, self.precomp_buf, self.precomp)

        # Compile the kernel
        self.program = cl.Program(self.ctx, kernel).build()

        return
        # return program

    @timeit
    def run_kernel(self, addresses, pub_keys):

        for ind, pubk in enumerate(pub_keys):
            # Point multiplication for target public key (can't do negative point addition for 1 << 192 * G in opencl with uint
            # so we prepare data in python)
            pub_key = self.sub_key(pubk[2:])
            x = [int(pub_key[i * 8:i * 8 + 8], 16) for i in range(8)][::-1]
            y = [int(pub_key[i * 8:i * 8 + 8], 16) for i in range(8, 16)][::-1]
            pubkey = np.empty(1, dtype=self.point_dtype)
            pubkey["x"] = np.array(x, dtype=np.uint32)
            pubkey["y"] = np.array(y, dtype=np.uint32)

            # Create head pointer buffer
            # Define the struct type
            struct_type = np.dtype([('x', np.uint32), ('y', np.uint32), ('z', np.uint32), ('w', np.uint32)])
            array_size = 50
            # Allocate memory for the array on the device
            # array_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, array_size * struct_type.itemsize)
            struct_array = np.array([(0, 0, 0, 0) for _ in range(array_size)], dtype=struct_type)

            # Create a buffer on the device to hold the struct array
            array_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, struct_array.nbytes)

            # Copy the struct array from the host to the device
            cl.enqueue_copy(self.queue, array_buf, struct_array)

            print('Starting kernel')
            print('Address: ', addresses[ind])
            print('Public key: ', pubk)
            t = time.time()
            self.program.kernel2(self.queue,
                                 self.global_size,
                                 self.local_size,
                                 self.precomp_buf,
                                 pubkey,
                                 self.set_buf,
                                 ctypes.c_uint32(self.iterations),
                                 ctypes.c_int32(len(self.elems)),
                                 array_buf)
            print('Time: ', time.time() - t, ' Global size: ', self.global_size)

            result_array = np.empty(array_size, dtype=struct_type)
            cl.enqueue_copy(self.queue, result_array, array_buf)

            self.queue.finish()
            res = list(set(result_array.tolist()))
            res.remove(tuple([0, 0, 0, 0]))
            try:
                session = self.Session()
                for r in res:
                    session.add(self.Collision(pk_hash=int(hashlib.md5(
                        bytes(str([r] + [self.start_percent, self.end_percent, addresses[ind]]),
                              'utf8')).hexdigest(), 16) % 2 ** 63,
                                               address=addresses[ind],
                                               public_key=pub_key,
                                               start_procentage=self.start_percent,
                                               end_procentage=self.end_percent,
                                               id_run=r[0],
                                               iteration=r[1],
                                               target_1=r[2],
                                               target_2=r[3]))
                    session.commit()
                session.add(
                    self.Collision(pk_hash=int(hashlib.md5(
                        bytes(str([self.start_percent, self.end_percent, addresses[ind]]),
                              'utf8')).hexdigest(), 16) % 2 ** 63,
                                   address=addresses[ind],
                                   public_key=pub_key,
                                   start_procentage=self.start_percent,
                                   end_procentage=self.end_percent,
                                   id_run=0,
                                   iteration=0,
                                   target_1=0,
                                   target_2=0))
                session.commit()
                session.close()
                print('Pct: ', round(ind / len(addresses), 2) * 100, '%')
            except Exception as e:
                print(e)
            self.check_collisions(pub_key)


